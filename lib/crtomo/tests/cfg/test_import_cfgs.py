#!/usr/bin/env python
# test importing crmod.cfg and crtomo.cfg from existing files into the
# corresponding cfg objects
from crtomo.cfg import crtomo_config


def test_crtomo_cfg_import_from_file():
    cfg = crtomo_config()
    cfg.import_from_file('input_files/crtomo.cfg')
    print(cfg)


if __name__ == '__main__':
    test_crtomo_cfg_import_from_file()
