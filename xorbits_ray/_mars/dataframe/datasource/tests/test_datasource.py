# Copyright 2022-2023 XProbe Inc.
# derived from copyright 1999-2021 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import pytest
import ray

from xorbits._mars.core import tile
from xorbits._mars.dataframe.utils import ray_deprecate_ml_dataset

from ..read_raydataset import (
    DataFrameReadMLDataset,
    DataFrameReadRayDataset,
    read_ray_dataset,
    read_ray_mldataset,
)

def test_read_ray_dataset(ray_start_regular):
    test_df1 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )
    test_df2 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )
    df = pd.concat([test_df1, test_df2])
    ds = ray.data.from_pandas_refs([ray.put(test_df1), ray.put(test_df2)])
    mdf = read_ray_dataset(ds)

    assert mdf.shape[1] == 2
    pd.testing.assert_index_equal(df.columns, mdf.columns_value.to_pandas())
    pd.testing.assert_series_equal(df.dtypes, mdf.dtypes)

    mdf = tile(mdf)
    assert len(mdf.chunks) == 2
    for chunk in mdf.chunks:
        assert isinstance(chunk.op, DataFrameReadRayDataset)


@pytest.mark.skipif(
    ray_deprecate_ml_dataset in (True, None),
    reason="Ray (>=2.0) has deprecated MLDataset.",
)
def test_read_ray_mldataset(ray_start_regular):
    test_df1 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )
    test_df2 = pd.DataFrame(
        {
            "a": np.arange(10).astype(np.int64, copy=False),
            "b": [f"s{i}" for i in range(10)],
        }
    )
    df = pd.concat([test_df1, test_df2])
    import ray.util.iter
    from ray.util.data import from_parallel_iter

    ml_dataset = from_parallel_iter(
        ray.util.iter.from_items([test_df1, test_df2], num_shards=2), need_convert=False
    )
    mdf = read_ray_mldataset(ml_dataset)

    assert mdf.shape[1] == 2
    pd.testing.assert_index_equal(df.columns, mdf.columns_value.to_pandas())
    pd.testing.assert_series_equal(df.dtypes, mdf.dtypes)

    mdf = tile(mdf)
    assert len(mdf.chunks) == 2
    for chunk in mdf.chunks:
        assert isinstance(chunk.op, DataFrameReadMLDataset)
