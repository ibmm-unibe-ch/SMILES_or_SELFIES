# Starting point of ring also before ring?
from our_representation import translate_to_own


# Wikipedia examples
def test_wikipedia_examples():
    assert translate_to_own("C2CCC1CCCCC1C2", False) == "<0;CCCCCC>?{3,4}<0;CCCCCC>!"
    assert translate_to_own("CN=C=O", False) == "CN=C=O"
    assert translate_to_own("COc1cc(C=O)ccc1O", False) == "CO<0;cccccc>?{2}cC=O!cO"
    assert (
        translate_to_own("COc2ccc1[nH]cc(CCNC(C)=O)c1c2", False)
        == "CO<0;cccccc>?{3,4}<2;cccc[nH]>?{1}cCCNC(CC)C=O!!"
    )
    assert (
        translate_to_own("CCc4ccc3c2[nH]c1ccccc1c2cc[n+]3c4", False)
        == "CC<5;ccccc[n+]>?{4,5}<0;ccccc[n+]>?{1,2}<2;cccc[nH]>?{0,1}<0;cccccc>!!!"
    )
    assert (
        translate_to_own("CN1CCC[C@H]1c2cccnc2", False)
        == "C<4;[C@H]NCCC>[C@H]<4;nccccc>"
    )
    assert (
        translate_to_own("CCC[C@@H](O)CC/C=C/C=C/C#CC#C/C=C/CO", False)
        == "CCC[C@@H]([C@@H]O)[C@@H]CC/C=C/C=C/C#CC#C/C=C/CO"
    )
    assert (
        translate_to_own(
            "C=C/C=C\CC2=C(C)[C@@H](OC(=O)[C@@H]1[C@@H](/C=C(C)/C(=O)OC)C1(C)C)CC2=O",
            False,
        )
        == "C=C/C=C\C<2;[C@@H]CCC=C>?{0}[C@@H]OC(C=O)C<0;[C@@H][C@@H]C>?{1}[C@@H]/C=C(CC)C/C(C=O)COC!C(CC)CC!?{4}CC!C=O"
    )
    assert (
        translate_to_own("COc4cc2O[C@H]1OC=C[C@H]1c2c5oc(=O)c3C(=O)CCc3c45", False)
        == "CO<0;cccccc>?{2,3}<4;ccO[C@H][C@H]>?{3,4}<4;[C@H][C@H]OC=C>!!?{4,5}<1;occccc>?{1}c=O!?{2,3}<4;ccCCC>?{2}C=O!!!"
    )
    assert (
        translate_to_own("OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O", False)
        == "OC<3;[C@H][C@@H][C@@H][C@H]O[C@@H]>?{0}[C@H]O!?{1}[C@@H]O!?{5}[C@@H]O![C@@H]O"
    )
    assert (
        translate_to_own(
            "COc3c(O)cc2C(=O)O[C@@H]1[C@@H](O)[C@H](O)[C@@H](CO)O[C@H]1c2c3O", False
        )
        == "CO<0;cccccc>?{1}cO!?{3,4}<5;ccCO[C@@H][C@H]>?{2}C=O!?{4,5}<5;[C@H][C@@H][C@@H][C@H][C@@H]O>?{2}[C@@H]O!?{3}[C@H]O!?{4}[C@@H]CO!!!cO"
    )
    assert (
        translate_to_own("C=CCC[C@H](C/C=C(C)\CCOC(C)=O)C(=C)C", False)
        == "C=CCC[C@H]([C@H]C/C=C(CC)C\CCOC(CC)C=O)[C@H]C(C=C)CC"
    )
    assert (
        translate_to_own("CC(C)[C@]12CC(=O)[C@H](C)[C@H]1C2", False)
        == "CC(CC)C<0;[C@][C@H]C>?{0,1}<0;[C@]CC[C@H][C@H]>?{2}C=O!?{3}[C@H]C!!"
    )
    assert (
        translate_to_own("Cc2ncc(C[n+]1csc(CCO)c1C)c(N)n2", False)
        == "C<5;ncnccc>?{4}cC<2;scc[n+]c>?{1}cCCO!cC!?{5}cN!"
    )


# Ring ordering
def test_ring_ordering():
    assert translate_to_own("F12CC1F1CC21") == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    assert translate_to_own("F12CC1F1CC12") == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    assert translate_to_own("F21CC1F1CC21") == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    assert translate_to_own("F21CC1F1CC12") == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"


# Ring ordering and % test
def test_ring_ordering_and_perc():
    assert (
        translate_to_own("F%10%22CC%10F%10CC%22%10", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        translate_to_own("F%10%22CC%10F%10CC%10%22", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        translate_to_own("F%22%10CC%10F%10CC%22%10", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        translate_to_own("F%22%10CC%10F%10CC%10%22", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )


# bond in branch test
def test_ring_in_branch():
    assert translate_to_own("FC1F[Cl](O1CC)F", False) == "F(<2;[Cl]OCF>CC)[Cl]F"


# hard SMILES
def test_hard_smiles():
    assert (
        translate_to_own("N#Cc1c(Oc2cccc(N)c2)ccc2nc(NC(=O)C3CC3)sc12", False)
        == "N#C<0;cccccc>?{1}cO<0;cccccc>?{4}cN!!?{4,5}<3;sccnc>?{4}cNC(C=O)C<0;CCC>!!"
    )
    assert (
        translate_to_own(
            "C/C=C/CC(C)C(O)C1C(=O)NC(CC)C(=O)N(C)CC(=O)N(C)C(CC(C)C)C(=O)NC(C(C)C)C(=O)N(C)C(CC(C)C)C(=O)NC(C)C(=O)NC(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(C(C)C)C(=O)N1C",
            False,
        )
        == "C/C=C/CC(CC)CC(CO)C<2;NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCNCC>?{1}CCC!?{2}C=O!?{3}NC!?{5}C=O!?{6}NC!?{7}CCC(CC)CC!?{8}C=O!?{10}CC(CC)CC!?{11}C=O!?{12}NC!?{13}CCC(CC)CC!?{14}C=O!?{16}CC!?{17}C=O!?{19}CC!?{20}C=O!?{21}NC!?{22}CCC(CC)CC!?{23}C=O!?{24}NC!?{25}CCC(CC)CC!?{26}C=O!?{27}NC!?{28}CC(CC)CC!?{29}C=O!?{32}C=O!NC"
    )
    assert (
        translate_to_own(
            "CC[C@H]3OC(=O)[C@H](C)[C@@H](O[C@H]1C[C@@](C)(OC)[C@@H](O)[C@H](C)O1)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]3(C)O",
            False,
        )
        == "CC<13;[C@][C@H]OC[C@H][C@@H][C@H][C@@H][C@]C[C@@H]C[C@H][C@@H]>?{3}C=O!?{4}[C@H]C!?{5}[C@@H]O<4;[C@H]O[C@H]C[C@@][C@@H]>?{0}[C@H]C!?{4}[C@@]C!?{4}[C@@]OC!?{5}[C@@H]O!!?{6}[C@H]C!?{7}[C@@H]O<4;[C@H][C@H][C@@H]O[C@H]C>?{0}[C@H]N(NC)NC!?{4}[C@H]C![C@H]O!?{8}[C@]C!?{8}[C@]O!?{10}[C@@H]C!?{11}C=O!?{12}[C@H]C!?{13}[C@@H]O![C@]([C@]C)[C@]O"
    )
