def validate_connectome(connectome, valid_regions=None, require_bidirectional=False):
    """
    Validates the connectome dictionary structure, bidirectionality, and region existence.

    Args:
        connectome (dict): The connectome data dictionary.
        valid_regions (set or list, optional): Known valid region names to check existence.
        require_bidirectional (bool): If True, require reciprocal connections.

    Checks performed:
    - Basic structure and parameter sanity.
    - Bidirectionality consistency if requested.
    - Referenced regions exist in valid_regions if provided.
    """

    errors = []
    warnings = []

    # Basic structure and parameter checks
    for src, targets in connectome.items():
        if not isinstance(src, str):
            errors.append(f"Source region name is not a string: {src} ({type(src)})")

        if valid_regions and src not in valid_regions:
            warnings.append(f"Source region '{src}' not in known valid region list.")

        if not isinstance(targets, dict):
            errors.append(f"Targets for source '{src}' is not a dict: {type(targets)}")
            continue

        for tgt, params in targets.items():
            if not isinstance(tgt, str):
                errors.append(f"Target region name is not a string: {tgt} ({type(tgt)})")

            if valid_regions and tgt not in valid_regions:
                warnings.append(f"Target region '{tgt}' not in known valid region list.")

            if not isinstance(params, dict):
                errors.append(f"Params for connection {src} -> {tgt} not a dict: {type(params)}")
                continue

            required_keys = ['weight', 'plasticity', 'delay_jitter']
            for key in required_keys:
                if key not in params:
                    errors.append(f"Missing key '{key}' for connection {src} -> {tgt}")

            weight = params.get('weight')
            plasticity = params.get('plasticity')
            delay_jitter = params.get('delay_jitter')

            if not isinstance(weight, (int, float)):
                errors.append(f"Weight not numeric for connection {src} -> {tgt}: {weight}")
            elif weight == 0:
                warnings.append(f"Weight is zero for connection {src} -> {tgt}")

            if not isinstance(plasticity, (int, float)):
                errors.append(f"Plasticity not numeric for connection {src} -> {tgt}: {plasticity}")
            elif plasticity <= 0:
                warnings.append(f"Plasticity <= 0 for connection {src} -> {tgt}: {plasticity}")

            if not isinstance(delay_jitter, (int, float)):
                errors.append(f"Delay jitter not numeric for connection {src} -> {tgt}: {delay_jitter}")
            elif delay_jitter < 0:
                warnings.append(f"Negative delay jitter for connection {src} -> {tgt}: {delay_jitter}")

    # Bidirectionality check
    if require_bidirectional:
        for src, targets in connectome.items():
            for tgt in targets.keys():
                reciprocal = connectome.get(tgt, {}).get(src, None)
                if reciprocal is None:
                    warnings.append(f"Missing reciprocal connection: {tgt} -> {src}")

    # Summary output
    if errors:
        print("Connectome validation errors found:")
        for err in errors:
            print("  ERROR:", err)
    else:
        print("No connectome validation errors found.")

    if warnings:
        print("Connectome validation warnings:")
        for warn in warnings:
            print("  WARNING:", warn)
    else:
        print("No connectome validation warnings.")

    print(f"Total regions (sources): {len(connectome)}")
    total_connections = sum(len(t) for t in connectome.values())
    print(f"Total connections defined: {total_connections}")
