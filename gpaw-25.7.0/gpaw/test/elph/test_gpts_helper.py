from gpaw.elph.gpts import main


def test_gpts(gpw_files, capsys):
    main(f'-g 0.2 -e 340 -s 2,2,2 {gpw_files["bcc_li_fd"]}'.split())
    out = capsys.readouterr().out
    assert 'Add "gpts=[24, 24, 24]" to PW mode calculator' in out
