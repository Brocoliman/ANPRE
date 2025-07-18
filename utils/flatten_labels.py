# ═════════════════════  helper: flatten field_labels.json tree  ════════════════


def flatten_labels(label_tree, blanks=True):
    flat = []
    for category, tests in label_tree.items():
        if blanks:
            flat.append((category, "", ""))
        else:
            flat.append((category,))
        if isinstance(tests, list):
            for t in tests:
                if blanks:
                    flat.append((category, t, ""))
                else:
                    flat.append((category, t))
        else:  # dict
            for test_name, subs in tests.items():
                if subs:
                    for sub in subs:
                        flat.append((category, test_name, sub))
                else:
                    if blanks:
                        flat.append((category, test_name, ""))
                    else:
                        flat.append((category, test_name))
    return flat
