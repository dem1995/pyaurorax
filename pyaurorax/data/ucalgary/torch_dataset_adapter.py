try:
    from torch.utils.data import Dataset
except ImportError:
    raise ImportError("PyTorch is not installed. Please install PyTorch to use this module.")
try:
    import h5py
except ImportError:
    raise ImportError("h5py is not installed. Please install h5py to use this module.")
import sys
if sys.version_info > (3, 10):
    from typing_extensions import Self as _SelfAuroraDataset, Self as _SelfAugmentedAuroraDataset
else:
    from typing import TypeVar
    _SelfAuroraDataset = TypeVar('_SelfAuroraDataset', bound='AuroraDataset')
    _SelfAugmentedAuroraDataset = TypeVar('_SelfAugmentedAuroraDataset', bound='AugmentedAuroraDataset')
from typing import Any, Callable, List, Optional, Sequence, Union, cast
from pathlib import Path
from collections import OrderedDict
import numpy as np
import datetime
from pyaurorax import pyaurorax


class AuroraTorchDataset(Dataset):
    """
    A PyTorch Dataset wrapper around a PyAuroraX download that only includes the image data.
    Use `AugmentedAuroraDataset` for a version that also includes metadata.
    """

    def __init__(
        self,
        file_paths: Sequence[Union[str, Path]],
        key: str = 'data/images',
        transform: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None
    ) -> None:
        self.file_paths: List[str] = [str(fp) for fp in file_paths]
        self.key: str = key
        self.h5_files: List[Optional[h5py.File]] = (
            [None] * len(self.file_paths)
        )
        self.open_file_indices: OrderedDict[int, None] = OrderedDict()  # For LRU caching

        # Get system file limit and set max open files safely
        # soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        soft_limit = 1024
        self.max_open_files: int = min(
            len(self.file_paths), max(10, soft_limit // 2)
        )

        # Determine how many samples are in each file.
        self.lengths: List[int] = []
        for path in self.file_paths:
            with h5py.File(path, 'r') as f:
                dset = f[self.key]
                if not isinstance(dset, h5py.Dataset):
                    raise ValueError(f"Dataset at {path} is not a valid h5py.Dataset.")
                if dset.ndim == 4:
                    self.lengths.append(dset.shape[-1])
                    # print(f"Dataset at {path} has {dset.shape} ({dset.shape[-1]}) samples.")
                else:
                    self.lengths.append(len(dset))
                    # print(f"Dataset at {path} has {len(dset)} samples.")

        cumulative_lengths: np.ndarray = np.cumsum([0] + self.lengths)
        self.cumulative_lengths: List[int] = cast(
            List[int], cumulative_lengths.tolist()
        )
        self.total_len: int = self.cumulative_lengths[-1]
        self.transform: Optional[Callable] = transform

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, index: int) -> torch.Tensor:
        file_idx, local_index = self._resolve_index(index)
        dset = self._get_dataset(file_idx)
        if dset.ndim == 4:
            sample = dset[..., local_index]
        else:
            sample = dset[local_index]

        if self.transform:
            sample = self.transform(sample)
        return torch.tensor(sample)

    def __getitems__(self, indices: Sequence[int]) -> List[torch.Tensor]:
        file_indices = [self._resolve_index(idx) for idx in indices]
        samples: List[Optional[torch.Tensor]] = [None] * len(indices)

        for i, (file_idx, local_idx) in enumerate(file_indices):
            dset = self._get_dataset(file_idx)
            if dset.ndim == 4:
                sample = dset[..., local_idx]
            else:
                sample = dset[local_idx]
            samples[i] = torch.tensor(sample)
            if self.transform:
                samples[i] = self.transform(samples[i])

        return cast(List[torch.Tensor], samples)

    def _resolve_index(self, index: int) -> tuple[np.signedinteger, int]:
        file_idx = np.searchsorted(self.cumulative_lengths, index, side='right') - 1
        local_index = index - self.cumulative_lengths[file_idx]
        return file_idx, local_index

    def _get_dataset(self, file_idx: int | np.signedinteger) -> h5py.Dataset:
        file_idx = int(file_idx)  # Convert to int

        if self.h5_files[file_idx] is None:
            self._maybe_close_files_to_free_space(exclude={file_idx})
            self.h5_files[file_idx] = h5py.File(self.file_paths[file_idx], 'r')
        # Update LRU order
        self.open_file_indices.pop(file_idx, None)
        self.open_file_indices[file_idx] = None

        file = self.h5_files[file_idx]
        if file is None:
            raise RuntimeError(f"File {self.file_paths[file_idx]} failed to open.")
        dset = file[self.key]
        if not isinstance(dset, h5py.Dataset):
            raise ValueError(f"{self.key} in {self.file_paths[file_idx]} is not a valid h5py.Dataset.")
        return dset

    def _maybe_close_files_to_free_space(self, exclude: set[int] = set()) -> None:
        while len(self.open_file_indices) >= self.max_open_files:
            lru_idx, _ = self.open_file_indices.popitem(last=False)
            if lru_idx in exclude:
                self.open_file_indices[lru_idx] = None  # Put it back
                continue

            self_h5_file_at_idx = self.h5_files[lru_idx]
            if self_h5_file_at_idx is not None:
                self_h5_file_at_idx.close()
                self.h5_files[lru_idx] = None

    def close(self) -> None:
        for i, f in enumerate(self.h5_files):
            if f is not None:
                f.close()
                self.h5_files[i] = None
        self.open_file_indices.clear()

    @classmethod
    def from_download_details(
        cls: type[_SelfAuroraDataset],
        start_date: datetime.datetime,
        end_date: datetime.datetime,
        dataset_name: str,
        site_uid: str,
        interval: str = 'all',
        interval_per_interval: str = '1m',
        transform: Optional[Callable] = None,
        download_root_path: Optional[str] = None
    ) -> _SelfAuroraDataset:
        """
        Downloads the TREX dataset from the U of Calgary and returns a dataset object.
        If the requested interval spans more than 24 hours, the download is split into
        within-date intervals (00:00:00 to 23:59:59 for full days) and concatenated.

        Args:
            start_date (datetime.datetime): Start date for data download.
            end_date (datetime.datetime): End date for data download.
            dataset_name (str): Name of the dataset to download.
            site_uid (str): Site UID for data download.
            download_root_path (Optional[str]): Path to download data. Defaults to None.

        Returns:
            AuroraDataset: Dataset object referencing the downloaded data.
        """
        if download_root_path:
            aurorax = pyaurorax.PyAuroraX(download_output_root_path=download_root_path)
        else:
            aurorax = pyaurorax.PyAuroraX()

        file_paths: List[str] = []

        if interval == 'all':
            current_start = start_date
            # Loop until the entire interval has been processed
            while current_start < end_date:
                # Set the end of the current interval to the end of the day
                end_of_day = current_start.replace(hour=23, minute=59, second=59, microsecond=0)
                # Ensure we do not exceed the overall end_date
                current_end = min(end_date, end_of_day)

                # Download the current interval
                r = aurorax.data.ucalgary.download(
                    dataset_name,
                    current_start,
                    current_end,
                    site_uid=site_uid
                )
                file_paths.extend(r.filenames)

                # Move to the start of the next day (00:00:00)
                next_day = current_start + datetime.timedelta(days=1)
                current_start = next_day.replace(hour=0, minute=0, second=0, microsecond=0)

        else:
            def convert_interval_to_datetime(interval: str | datetime.datetime) -> datetime.timedelta:
                """
                Converts a time interval string to a datetime.timedelta object.
                """
                # If it's already hh:mm:ss format, parse it directly
                if isinstance(interval, datetime.timedelta):
                    return interval

                elif isinstance(interval, datetime.datetime):
                    return datetime.timedelta(hours=interval.hour, minutes=interval.minute, seconds=interval.second)
                
                elif isinstance(interval, str):
                    if ':' in interval:
                        parts = list(map(int, interval.split(':')))
                        return datetime.timedelta(hours=parts[0], minutes=parts[1], seconds=parts[2])
                    elif interval.endswith('h'):
                        return datetime.timedelta(hours=int(interval[:-1]))
                    elif interval.endswith('m'):
                        return datetime.timedelta(minutes=int(interval[:-1]))
                    elif interval.endswith('s'):
                        return datetime.timedelta(seconds=int(interval[:-1]))
                    else:
                        raise ValueError(f"Unsupported interval format: {interval}")

            interval_timedelta = convert_interval_to_datetime(interval)
            interval_per_interval_timedelta = convert_interval_to_datetime(interval_per_interval)

            current_mid = start_date
            current_start = current_mid - interval_per_interval_timedelta
            current_end = current_mid + interval_per_interval_timedelta

            # Loop until the entire interval has been processed
            while current_mid <= end_date:
                # Ensure we do not exceed the overall end_date
                # current_end = min(end_date, current_start + interval_timedelta)

                # Download the current interval
                r = aurorax.data.ucalgary.download(
                    dataset_name,
                    current_start,
                    current_end,
                    site_uid=site_uid
                )
                file_paths.extend(r.filenames)

                # Move to the start of the next interval
                current_mid = current_mid + interval_timedelta
                current_start = current_mid - interval_per_interval_timedelta
                current_end = current_mid + interval_per_interval_timedelta

        return cls(file_paths, transform=transform)

    def __del__(self) -> None:
        self.close()


class AugmentedAuroraTorchDataset(Dataset):
    """
    A PyTorch Dataset wrapper around a PyAuroraX download that additionally includes metadata.
    """

    def __init__(self, aurora_dataset: AuroraTorchDataset) -> None:
        self.dataset = aurora_dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        # Get the image tensor via the wrapped AuroraDataset.
        image_tensor = self.dataset[index]

        # Retrieve internal indices needed to extract metadata.
        file_idx, local_index = self.dataset._resolve_index(index)
        file = self.dataset.h5_files[file_idx]
        if file is None:
            # Open the file if not already open.
            self.dataset._maybe_close_files_to_free_space(exclude={file_idx})
            file = h5py.File(self.dataset.file_paths[file_idx], 'r')
            self.dataset.h5_files[file_idx] = file

        metadata: dict[str, Any] = {}

        # Get file-level metadata from 'metadata/file'.
        meta_group = file.get('metadata')
        assert isinstance(meta_group, h5py.Group)
        if meta_group is not None:
            file_meta = meta_group.get('file')
            # print(type(file_meta))
            # print(file_meta.__dict__)
            assert isinstance(file_meta, h5py.Dataset)
            metadata['file'] = dict(file_meta.attrs) if file_meta is not None else {}
        else:
            metadata['file'] = {}

        # Get frame-level metadata from 'metadata/frame/frame{N}'.
        if meta_group is not None:
            frame_group = meta_group.get('frame')
            assert isinstance(frame_group, h5py.Group)
            if frame_group is not None:
                frame_key = f'frame{local_index}'
                frame_meta = frame_group.get(frame_key)
                assert isinstance(frame_meta, h5py.Dataset)
                metadata['frame'] = dict(frame_meta.attrs) if frame_meta is not None else {}
            else:
                metadata['frame'] = {}
        else:
            metadata['frame'] = {}

        # Optionally, attach timestamp information from 'data/timestamp'.
        data_group = file.get('data')
        assert isinstance(data_group, h5py.Group)
        if data_group is not None and 'timestamp' in data_group:
            timestamp_dset = data_group['timestamp']
            assert isinstance(timestamp_dset, h5py.Dataset)
            try:
                metadata['timestamp'] = timestamp_dset[local_index]
            except IndexError:
                metadata['timestamp'] = None

        return image_tensor, metadata

    # def to_lazy_xarray_dataset(self) -> xr.Dataset:
    #     dataset = self
    #     print(len(dataset))

    #     @delayed
    #     def load_sample(idx):
    #         tensor, meta = dataset[idx]
    #         tensor_np = np.array(tensor)

    #         metadata_fields_file: dict = meta['file']
    #         metadata_fields_frame: dict = meta['frame']

    #         all_metadata_fields = {str(key): str(value) for key, value
    #                                in metadata_fields_file.items()}
    #         all_metadata_fields.update({str(key): str(value) for key, value in
    #                                     metadata_fields_frame.items()})

    #         return tensor_np, all_metadata_fields

    #     delayed_results = [load_sample(i) for i in range(len(dataset))]
    #     delayed_images = [res.map(lambda x: x[0]) for res in delayed_results]
    #     delayed_metadata = [res.map(lambda x: x[1]) for res in delayed_results]

    #     exemplar_sample = delayed_results[0].compute()
    #     print(type(delayed_results[0]))
    #     exemplar_metadata = exemplar_sample[1]

    #     lazy_images = da.stack(
    #         [da.from_delayed(res, shape=exemplar_sample[0].shape, dtype=exemplar_sample[0].dtype) for res in delayed_images],
    #         axis=0).rechunk(chunks=(32, *exemplar_sample[0].shape))

    #     metadata_keys = list(exemplar_metadata.keys())
    #     metadata_fields = {
    #         key: [dm.map(lambda d, k=key: d[k]) for dm in delayed_metadata]
    #         for key in metadata_keys
    #     }
    #     print(type(delayed_results[0]))

    #     lazy_metadata_stacks = {
    #         key: da.stack(
    #             [da.from_delayed(res, shape=(), dtype=np.dtype(type(value))) for res in metadata_fields[key]], axis=0
    #         ).rechunk(chunks=(32,))
    #         for key, value in metadata_fields.items()
    #     }

    #     coords = {"sample": np.arange(len(delayed_images))}
    #     coords.update({
    #         key: (("sample",), lazy_metadata_stacks[key]) for key in lazy_metadata_stacks.keys()
    #     })

    #     print(type(delayed_results[0]))
    #     # exit()

    #     image_da = xr.DataArray(
    #         lazy_images,
    #         dims=("sample", "y", "x", "channel"),
    #         coords=coords,
    #     )

    #     ds = xr.Dataset({"image": image_da})

    #     return ds


    #     for i in range(len(self.dataset)):
    #         tensor, meta = load_sample(i)
    #         delayed_tensors.append(tensor)
    #         delayed_metadata.append(meta)
        
    #     How 


    # def to_lazy_xarray_dataset(self) -> xr.Dataset:
    #     dataset = self
    #     sample_tensor, sample_meta = dataset[0]  # load one sample to get shape/dtype
    #     img_shape = np.array(sample_tensor).shape  # e.g. (480, 553, 3)
    #     img_dtype = np.array(sample_tensor).dtype  # ensure NumPy dtype (e.g. np.uint8 or np.float32)


    #     @delayed
    #     def load_sample(idx):
    #         tensor, meta = dataset[idx]
    #         tensor_np = np.array(tensor)

    #         metadata_fields_file: dict = meta['file']
    #         metadata_fields_frame: dict = meta['frame']

    #         all_metadata_fields = {str(key): str(value) for key, value
    #                                in metadata_fields_file.items()}
    #         all_metadata_fields.update({str(key): str(value) for key, value in
    #                                     metadata_fields_frame.items()})

    #         return tensor_np, all_metadata_fields

    #     # lazily load all samples
    #     lazy_results = [load_sample(i) for i in range(len(dataset))]
    #     lazy_images = [da.from_delayed(res[0], shape=img_shape, dtype=img_dtype) for res in lazy_results]
    #     lazy_images = da.stack(lazy_images, axis=0)
    #     lazy_images = lazy_images.rechunk(chunks=(32, *img_shape))  # chunk 32 samples per chunk

    #     # lazy_metadata_stack_generator
    #     def get_lazy_metadata_stacks():
    #         example_sample = lazy_results[0].compute()
    #         example_metadata = example_sample[1]

    #         example_metadata = {str(key): str(value) for key, value in example_metadata.items()}
    #         metadata_shape = {str(key): () for key in example_metadata.keys()}
    #         metadata_dtypes = {str(key): np.dtype(type(value)) for key, value in example_metadata.items()}

    #         metadata_stacks = dict()

    #         for key, value in example_metadata.items():
    #             metadata_stacks[key] = da.stack(
    #                 [da.from_delayed(res, shape=metadata_shape[key], dtype=metadata_dtypes[key])
    #                  for res in lazy_results], axis=0
    #             )

    #         return metadata_stacks
        

    #     lazy_metadata_stacks = get_lazy_metadata_stacks()

    #     for key, value in lazy_metadata_stacks.items():
    #         lazy_metadata_stacks[key] = lazy_metadata_stacks[key].rechunk(chunks=(32, *value.shape))
        

    #     # Build XArray Dataset with images and metadata variables
    #     # The metadata should be assigned as coordinates in the dataset
    #     # using lazy_metadata_stacks as coordinates
    #     ds_b = xr.Dataset(
    #         data_vars={
    #             "image": (("sample", "y", "x", "channel"), lazy_images),
    #         },
    #         coords= {
    #             key: (("sample",), lazy_metadata_stacks[key]) for key in lazy_metadata_stacks.keys()
    #         }
                
    #     )
    #     # Add metadata variables to the dataset
    #     for key, value in lazy_metadata_stacks.items():
    #         ds_b[key] = (("sample",), value)
    #         ds_b[key].attrs['description'] = f"Metadata for {key}"






    # def to_lazy_xarray_dataset(self) -> xr.Dataset:
    #     # Suppose dataset is your PyTorch Dataset instance
    #     dataset = self

    #     # Get dataset length and a sample for shape/dtype
    #     N = len(dataset)
    #     sample_img, sample_meta = dataset[0]  # load one sample to get shape/dtype
    #     data_shape = tuple(sample_img.shape)         # e.g. (480, 553, 3)
    #     data_dtype = np.array(sample_img).dtype      # ensure NumPy dtype (e.g., np.uint8 or np.float32)

    #     # Define a function to lazily load one image (as NumPy array)
    #     def load_image(idx):
    #         tensor, meta = dataset[idx]             # get one sample
    #         return np.array(tensor)                # convert torch.Tensor to numpy array

    #     # Create a list of delayed load tasks, one per sample
    #     lazy_loads = [delayed(load_image)(i) for i in range(N)]

    #     # Wrap each delayed result as a single-chunk Dask array
    #     dask_arrays = [da.from_delayed(obj, shape=data_shape, dtype=data_dtype) 
    #                 for obj in lazy_loads]

    #     # Stack Dask arrays along a new dimension (the sample dimension)
    #     images_dask = da.stack(dask_arrays, axis=0)

    #     # Dask-delayed function to load full metadata dict for one sample
    #     def load_metadata(idx):
    #         _, meta = dataset[idx]
    #         return meta  # return the metadata dictionary as is

    #     # Create a list of delayed metadata objects for each sample
    #     metadata_delayed = [delayed(load_metadata)(i) for i in range(N)]

    #     # Wrap each delayed metadata dict into a 0d dask array (object dtype)
    #     metadata_arrays = [da.from_delayed(md, shape=(), dtype=object) 
    #                     for md in metadata_delayed]

    #     # Stack into one Dask array of shape (N,) with object dtype
    #     metadata_dask = da.stack(metadata_arrays, axis=0)
    #     metadata_dask = metadata_dask.rechunk({0: 32})   # chunk 32 samples per chunk

    #     # Build Xarray Dataset with images and a single 'metadata' variable
    #     ds_b = xr.Dataset(
    #         data_vars={
    #             "image": (("sample", "y", "x", "channel"), images_dask),  # reuse images_dask from above
    #             "metadata": (("sample",), metadata_dask)
    #         }
    #     )

    #     return ds_b

    #     # Set chunking along the sample dimension (e.g., chunks of 32 samples)
    #     images_dask = images_dask.rechunk(chunks=(32, *data_shape))


    @classmethod
    def from_download_details(cls: type[_SelfAugmentedAuroraDataset], *args, **kwargs) -> _SelfAugmentedAuroraDataset:
        """
        Downloads the TREX dataset from the U of Calgary and returns an augmented dataset object.

        Args:
            start_date (datetime.datetime): Start date for data download.
            end_date (datetime.datetime): End date for data download.
            dataset_name (str): Name of the dataset to download.
            site_uid (str): Site UID for data download.
            download_root_path (Optional[str]): Path to download data. Defaults to None.

        Returns:
            AugmentedAuroraDataset: Augmented dataset object referencing the downloaded data.
        """

        base_dataset = AuroraTorchDataset.from_download_details(*args, **kwargs)
        return cls(base_dataset)
