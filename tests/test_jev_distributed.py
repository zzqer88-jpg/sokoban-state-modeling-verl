"""CPU checks of synchronized schedules and FSDP mean-gradient weighting."""
import torch
import pytest
from sokoban_state_modeling.jev.distributed import step_assignments, weighted_micro_loss, summarize_predictions
from sokoban_state_modeling.jev.scoring import cell_kl
from sokoban_state_modeling.jev.config import Config


@pytest.mark.parametrize('count', [1, 7, 8, 9, 17, 31])
@pytest.mark.parametrize('boards_per_rank', [1, 2])
def test_full_coverage_without_duplicate_real_boards(count, boards_per_rank):
    seen = []
    for batch, real in step_assignments(list(range(count)), 8, boards_per_rank):
        assert len(batch) == 8
        assert all(len(rank) == boards_per_rank for rank in batch)
        values = [i for rank in batch for i in rank if i is not None]
        assert len(values) == real
        seen.extend(values)
    assert sorted(seen) == list(range(count))


def test_uneven_boards_and_padding_match_global_board_mean_gradient():
    torch.manual_seed(13)
    weights = torch.randn(7, requires_grad=True)
    sizes = [4, 7, 3, 9, 2]  # Tail batch has fewer boards than ranks.
    inputs = [torch.randn(size, 7) for size in sizes]
    targets = [torch.nn.functional.one_hot(torch.arange(size) % 7, 7).float() for size in sizes]
    expected_loss = sum(cell_kl(x+weights, q).mean() for x, q in zip(inputs, targets))/len(sizes)
    expected = torch.autograd.grad(expected_loss, weights)[0]
    grads = []
    for rank in range(8):
        local_loss = weights.sum()*0
        index = rank if rank < len(sizes) else 0
        rounds = (sizes[index]+1)//2 if rank < len(sizes) else 0
        for micro in range(max((size+1)//2 for size in sizes)):
            valid = micro < rounds
            start = micro*2 if valid else 0
            end = min(start+2, sizes[index]) if valid else 1
            local_loss = local_loss + weighted_micro_loss(
                inputs[index][start:end]+weights, targets[index][start:end],
                board_cells=sizes[index], real_boards=len(sizes), world_size=8, valid=valid)
        grads.append(torch.autograd.grad(local_loss, weights)[0])
    torch.testing.assert_close(torch.stack(grads).mean(0), expected)


def test_aggregate_validation_and_fsdp_config():
    report = summarize_predictions([
        {'size': '1x2', 'truth': [5, 1], 'prediction': [5, 1]},
        {'size': '1x3', 'truth': [5, 3, 2], 'prediction': [5, 1, 2]},
    ])
    assert report['board_exact'] == .5
    assert report['cell_accuracy'] == .8
    assert report['boards'] == 2
    with pytest.raises(ValueError):
        Config(model_path='test', distributed='fsdp2', finetune_mode='lora')
