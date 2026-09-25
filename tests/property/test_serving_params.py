from hypothesis import given
from hypothesis import strategies as st

from localmodels.enums import DeviceType
from localmodels.params import derive_serving_params


@given(st.sampled_from(list(DeviceType)), st.floats(min_value=0.5, max_value=2048.0, allow_nan=False))
def test_params_within_bounds_and_monotone(device: DeviceType, mem: float) -> None:
    p = derive_serving_params(device, mem)
    assert 2048 <= p.n_ctx <= 32768
    assert 1 <= p.n_parallel <= 4
    bigger = derive_serving_params(device, mem * 2)
    assert bigger.n_ctx >= p.n_ctx
    assert bigger.n_parallel >= p.n_parallel
