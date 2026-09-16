from sokoban_state_modeling.prompt.parser import parse_response


GRID = ["######", "#____#", "#_PXO#", "#____#", "#____#", "######"]


def response(perception=GRID, prediction=GRID):
    return "<perception>\n" + "\n".join(perception) + "\n</perception>\n<prediction>\n" + "\n".join(prediction) + "\n</prediction>"


def test_exact_response_and_crlf():
    parsed = parse_response(response(), 6, 6)
    assert parsed.format_valid and parsed.format_exact
    parsed_crlf = parse_response(response().replace("\n", "\r\n"), 6, 6)
    assert parsed_crlf.format_exact


def test_sections_score_independently_but_extra_text_breaks_exact_format():
    parsed = parse_response("explanation\n" + response(), 6, 6)
    assert parsed.perception.syntax_valid and parsed.prediction.syntax_valid
    assert parsed.format_valid and not parsed.format_exact


def test_whitespace_inside_grid_is_invalid():
    bad = GRID.copy()
    bad[2] += " "
    parsed = parse_response(response(perception=bad), 6, 6)
    assert not parsed.perception.syntax_valid
    assert parsed.prediction.syntax_valid
