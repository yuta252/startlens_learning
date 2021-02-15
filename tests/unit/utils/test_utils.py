import pytest

from utils.utils import get_class_label_from_path


file_label_parameter = [
    ("/uplaoeds/picture/1/2/xxxxx.jpg", 2),
    ("/uplaods/picture/1/3/xxxxx.jpg", 3)
]


@pytest.mark.parametrize('file_path, label', file_label_parameter)
def test_get_class_label_from_path(file_path, label):
    """Correct case
    """
    assert get_class_label_from_path(file_path) == label


# @pytest.mark.skip(reason='skip!')
@pytest.mark.parametrize('file_path, label', [("/uplaoeds/picture/1/3/xxxxx.jpg", 2)])
def test_get_class_label_from_path_invalid(file_path, label, request):
    """Incorrect case: file path and class do not match
    """
    env = request.config.getoption('--env')
    print(env)
    assert not get_class_label_from_path(file_path) == label
