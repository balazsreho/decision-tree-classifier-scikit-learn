# pytest tests
from app.decisionTreeExample import prepare_training_data

def test_dummy():
    assert 1 == 1

def test_train_data_size():
    # no inputs, returns training output, test output
    # FIXME
    print(prepare_training_data())
    assert prepare_training_data() == ([[False, 0], [False, 1], [False, 2], [False, 3], [False, 4], [False, 5], [False, 6], [False, 7], [False, 8], [True, 0], [True, 1], [True, 2], [True, 3], [True, 4], [True, 5], [True, 6], [True, 7], [True, 8]], [False, False, False, True, False, False, False, False, False, False, False, True, True, False, False, False, False, False])