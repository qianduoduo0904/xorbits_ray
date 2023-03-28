
@require_ray
@pytest.mark.skip_ray_dag  # raydataset is not compatible with Ray DAG
def test_read_raydataset(ray_start_regular, ray_create_mars_cluster):
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
    mdf = md.read_ray_dataset(ds)
    assert df.equals(mdf.execute().fetch())

    n = 10000
    pdf = pd.DataFrame({"a": list(range(n)), "b": list(range(n, 2 * n))})
    df = md.DataFrame(pdf)

    # Convert mars dataframe to ray dataset
    ds = md.to_ray_dataset(df)
    pd.testing.assert_frame_equal(ds.to_pandas(), df.to_pandas())
    ds2 = ds.filter(lambda row: row["a"] % 2 == 0)
    assert ds2.take(5) == [{"a": 2 * i, "b": n + 2 * i} for i in range(5)]

    # Convert ray dataset to mars dataframe
    df2 = md.read_ray_dataset(ds2)
    pd.testing.assert_frame_equal(
        df2.head(5).to_pandas(),
        pd.DataFrame({"a": list(range(0, 10, 2)), "b": list(range(n, n + 10, 2))}),
    )

    # Test Arrow Dataset
    pdf2 = pd.DataFrame({c: range(5) for c in "abc"})
    ds3 = ray.data.from_arrow([pa.Table.from_pandas(pdf2) for _ in range(3)])
    df3 = md.read_ray_dataset(ds3)
    pd.testing.assert_frame_equal(
        df3.head(5).to_pandas(),
        pdf2,
    )


@require_ray
@pytest.mark.skipif(
    ray_deprecate_ml_dataset in (True, None),
    reason="Ray (>=2.0) has deprecated MLDataset.",
)
def test_read_ray_mldataset(ray_start_regular, ray_create_mars_cluster):
    test_dfs = [
        pd.DataFrame(
            {
                "a": np.arange(i * 10, (i + 1) * 10).astype(np.int64, copy=False),
                "b": [f"s{j}" for j in range(i * 10, (i + 1) * 10)],
            }
        )
        for i in range(5)
    ]
    import ray.util.iter
    from ray.util.data import from_parallel_iter

    ml_dataset = from_parallel_iter(
        ray.util.iter.from_items(test_dfs, num_shards=4), need_convert=False
    )
    dfs = []
    for shard in ml_dataset.shards():
        dfs.extend(list(shard))
    df = pd.concat(dfs).reset_index(drop=True)
    mdf = md.read_ray_mldataset(ml_dataset)
    pd.testing.assert_frame_equal(df, mdf.execute().fetch())
    pd.testing.assert_frame_equal(df.head(5), mdf.head(5).execute().fetch())
    pd.testing.assert_frame_equal(df.head(15), mdf.head(15).execute().fetch())
