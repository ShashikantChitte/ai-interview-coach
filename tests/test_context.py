from interview_coach.context import count_tokens, trim_messages


def conv(turns):
    msgs = [{"role": "system", "content": "SYS"}]
    for i in range(turns):
        msgs += [{"role": "user", "content": f"answer {i} " * 50},
                 {"role": "assistant", "content": "", "tool_calls": [{"id": f"c{i}"}]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": "{}"},
                 {"role": "assistant", "content": f"reply {i} " * 50}]
    return msgs


def test_no_trim_when_under_budget():
    msgs = conv(1)
    assert trim_messages(msgs, 10_000) == msgs


def test_trim_pins_system_and_latest_turn_and_keeps_tool_pairs():
    msgs = conv(6)
    out = trim_messages(msgs, count_tokens(msgs) // 2)
    assert out[0]["role"] == "system"
    assert out[-1] == msgs[-1]
    assert out[1]["role"] == "user"  # never starts mid-turn / with an orphan tool message
    assert len(out) < len(msgs)
    ids_called = {tc["id"] for m in out for tc in m.get("tool_calls", [])}
    ids_returned = {m["tool_call_id"] for m in out if m["role"] == "tool"}
    assert ids_called == ids_returned


def test_latest_turn_survives_tiny_budget():
    msgs = conv(3)
    out = trim_messages(msgs, 1)
    assert out == [msgs[0]] + msgs[-4:]
