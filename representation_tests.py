# Starting point of ring also before ring?
from our_representation import tokenise_our_representation, translate_to_own


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


def test_wikipedia_tokenisation():
    assert tokenise_our_representation("<0;CCCCCC>?{3,4}<0;CCCCCC>!") == [
        "<0;",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        ">",
        "?{",
        "3",
        ",",
        "4",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        ">",
        "!",
    ]
    assert tokenise_our_representation("CN=C=O") == ["C", "N", "=", "C", "=", "O"]
    assert tokenise_our_representation("CO<0;cccccc>?{2}cC=O!cO") == [
        "C",
        "O",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "2",
        "}",
        "c",
        "C",
        "=",
        "O",
        "!",
        "c",
        "O",
    ]
    assert tokenise_our_representation(
        "CO<0;cccccc>?{3,4}<2;cccc[nH]>?{1}cCCNC(CC)C=O!!"
    ) == [
        "C",
        "O",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "3",
        ",",
        "4",
        "}",
        "<2;",
        "c",
        "c",
        "c",
        "c",
        "[nH]",
        ">",
        "?{",
        "1",
        "}",
        "c",
        "C",
        "C",
        "N",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "=",
        "O",
        "!",
        "!",
    ]
    assert tokenise_our_representation(
        "CC<5;ccccc[n+]>?{4,5}<0;ccccc[n+]>?{1,2}<2;cccc[nH]>?{0,1}<0;cccccc>!!!"
    ) == [
        "C",
        "C",
        "<5;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "[n+]",
        ">",
        "?{",
        "4",
        ",",
        "5",
        "}",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "[n+]",
        ">",
        "?{",
        "1",
        ",",
        "2",
        "}",
        "<2;",
        "c",
        "c",
        "c",
        "c",
        "[nH]",
        ">",
        "?{",
        "0",
        ",",
        "1",
        "}",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "!",
        "!",
        "!",
    ]
    assert tokenise_our_representation("C<4;[C@H]NCCC>[C@H]<4;nccccc>") == [
        "C",
        "<4;",
        "[C@H]",
        "N",
        "C",
        "C",
        "C",
        ">",
        "[C@H]",
        "<4;",
        "n",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
    ]
    assert tokenise_our_representation(
        "CCC[C@@H]([C@@H]O)[C@@H]CC/C=C/C=C/C#CC#C/C=C/CO"
    ) == [
        "C",
        "C",
        "C",
        "[C@@H]",
        "(",
        "[C@@H]",
        "O",
        ")",
        "[C@@H]",
        "C",
        "C",
        "/",
        "C",
        "=",
        "C",
        "/",
        "C",
        "=",
        "C",
        "/",
        "C",
        "#",
        "C",
        "C",
        "#",
        "C",
        "/",
        "C",
        "=",
        "C",
        "/",
        "C",
        "O",
    ]
    assert tokenise_our_representation(
        "C=C/C=C\C<2;[C@@H]CCC=C>?{0}[C@@H]OC(C=O)C<0;[C@@H][C@@H]C>?{1}[C@@H]/C=C(CC)C/C(C=O)COC!C(CC)CC!?{4}CC!C=O"
    ) == [
        "C",
        "=",
        "C",
        "/",
        "C",
        "=",
        "C",
        "\\",
        "C",
        "<2;",
        "[C@@H]",
        "C",
        "C",
        "C",
        "=",
        "C",
        ">",
        "?{",
        "0",
        "}",
        "[C@@H]",
        "O",
        "C",
        "(",
        "C",
        "=",
        "O",
        ")",
        "C",
        "<0;",
        "[C@@H]",
        "[C@@H]",
        "C",
        ">",
        "?{",
        "1",
        "}",
        "[C@@H]",
        "/",
        "C",
        "=",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "/",
        "C",
        "(",
        "C",
        "=",
        "O",
        ")",
        "C",
        "O",
        "C",
        "!",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "4",
        "}",
        "C",
        "C",
        "!",
        "C",
        "=",
        "O",
    ]
    assert tokenise_our_representation(
        "CO<0;cccccc>?{2,3}<4;ccO[C@H][C@H]>?{3,4}<4;[C@H][C@H]OC=C>!!?{4,5}<1;occccc>?{1}c=O!?{2,3}<4;ccCCC>?{2}C=O!!!"
    ) == [
        "C",
        "O",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "2",
        ",",
        "3",
        "}",
        "<4;",
        "c",
        "c",
        "O",
        "[C@H]",
        "[C@H]",
        ">",
        "?{",
        "3",
        ",",
        "4",
        "}",
        "<4;",
        "[C@H]",
        "[C@H]",
        "O",
        "C",
        "=",
        "C",
        ">",
        "!",
        "!",
        "?{",
        "4",
        ",",
        "5",
        "}",
        "<1;",
        "o",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "1",
        "}",
        "c",
        "=",
        "O",
        "!",
        "?{",
        "2",
        ",",
        "3",
        "}",
        "<4;",
        "c",
        "c",
        "C",
        "C",
        "C",
        ">",
        "?{",
        "2",
        "}",
        "C",
        "=",
        "O",
        "!",
        "!",
        "!",
    ]
    assert tokenise_our_representation(
        "OC<3;[C@H][C@@H][C@@H][C@H]O[C@@H]>?{0}[C@H]O!?{1}[C@@H]O!?{5}[C@@H]O![C@@H]O"
    ) == [
        "O",
        "C",
        "<3;",
        "[C@H]",
        "[C@@H]",
        "[C@@H]",
        "[C@H]",
        "O",
        "[C@@H]",
        ">",
        "?{",
        "0",
        "}",
        "[C@H]",
        "O",
        "!",
        "?{",
        "1",
        "}",
        "[C@@H]",
        "O",
        "!",
        "?{",
        "5",
        "}",
        "[C@@H]",
        "O",
        "!",
        "[C@@H]",
        "O",
    ]
    assert tokenise_our_representation(
        "CO<0;cccccc>?{1}cO!?{3,4}<5;ccCO[C@@H][C@H]>?{2}C=O!?{4,5}<5;[C@H][C@@H][C@@H][C@H][C@@H]O>?{2}[C@@H]O!?{3}[C@H]O!?{4}[C@@H]CO!!!cO"
    ) == [
        "C",
        "O",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "1",
        "}",
        "c",
        "O",
        "!",
        "?{",
        "3",
        ",",
        "4",
        "}",
        "<5;",
        "c",
        "c",
        "C",
        "O",
        "[C@@H]",
        "[C@H]",
        ">",
        "?{",
        "2",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "4",
        ",",
        "5",
        "}",
        "<5;",
        "[C@H]",
        "[C@@H]",
        "[C@@H]",
        "[C@H]",
        "[C@@H]",
        "O",
        ">",
        "?{",
        "2",
        "}",
        "[C@@H]",
        "O",
        "!",
        "?{",
        "3",
        "}",
        "[C@H]",
        "O",
        "!",
        "?{",
        "4",
        "}",
        "[C@@H]",
        "C",
        "O",
        "!",
        "!",
        "!",
        "c",
        "O",
    ]
    assert tokenise_our_representation(
        "C=CCC[C@H]([C@H]C/C=C(CC)C\CCOC(CC)C=O)[C@H]C(C=C)CC"
    ) == [
        "C",
        "=",
        "C",
        "C",
        "C",
        "[C@H]",
        "(",
        "[C@H]",
        "C",
        "/",
        "C",
        "=",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "\\",
        "C",
        "C",
        "O",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "=",
        "O",
        ")",
        "[C@H]",
        "C",
        "(",
        "C",
        "=",
        "C",
        ")",
        "C",
        "C",
    ]
    assert tokenise_our_representation(
        "CC(CC)C<0;[C@][C@H]C>?{0,1}<0;[C@]CC[C@H][C@H]>?{2}C=O!?{3}[C@H]C!!"
    ) == [
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "<0;",
        "[C@]",
        "[C@H]",
        "C",
        ">",
        "?{",
        "0",
        ",",
        "1",
        "}",
        "<0;",
        "[C@]",
        "C",
        "C",
        "[C@H]",
        "[C@H]",
        ">",
        "?{",
        "2",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "3",
        "}",
        "[C@H]",
        "C",
        "!",
        "!",
    ]
    assert tokenise_our_representation(
        "C<5;ncnccc>?{4}cC<2;scc[n+]c>?{1}cCCO!cC!?{5}cN!"
    ) == [
        "C",
        "<5;",
        "n",
        "c",
        "n",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "4",
        "}",
        "c",
        "C",
        "<2;",
        "s",
        "c",
        "c",
        "[n+]",
        "c",
        ">",
        "?{",
        "1",
        "}",
        "c",
        "C",
        "C",
        "O",
        "!",
        "c",
        "C",
        "!",
        "?{",
        "5",
        "}",
        "c",
        "N",
        "!",
    ]


def test_wikipedia_tokenisation_no_drop():
    assert (
        "".join(tokenise_our_representation("<0;CCCCCC>?{3,4}<0;CCCCCC>!"))
        == "<0;CCCCCC>?{3,4}<0;CCCCCC>!"
    )
    assert "".join(tokenise_our_representation("CN=C=O")) == "CN=C=O"
    assert (
        "".join(tokenise_our_representation("CO<0;cccccc>?{2}cC=O!cO"))
        == "CO<0;cccccc>?{2}cC=O!cO"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CO<0;cccccc>?{3,4}<2;cccc[nH]>?{1}cCCNC(CC)C=O!!"
            )
        )
        == "CO<0;cccccc>?{3,4}<2;cccc[nH]>?{1}cCCNC(CC)C=O!!"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CC<5;ccccc[n+]>?{4,5}<0;ccccc[n+]>?{1,2}<2;cccc[nH]>?{0,1}<0;cccccc>!!!"
            )
        )
        == "CC<5;ccccc[n+]>?{4,5}<0;ccccc[n+]>?{1,2}<2;cccc[nH]>?{0,1}<0;cccccc>!!!"
    )
    assert (
        "".join(tokenise_our_representation("C<4;[C@H]NCCC>[C@H]<4;nccccc>"))
        == "C<4;[C@H]NCCC>[C@H]<4;nccccc>"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CCC[C@@H]([C@@H]O)[C@@H]CC/C=C/C=C/C#CC#C/C=C/CO"
            )
        )
        == "CCC[C@@H]([C@@H]O)[C@@H]CC/C=C/C=C/C#CC#C/C=C/CO"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "C=C/C=C\C<2;[C@@H]CCC=C>?{0}[C@@H]OC(C=O)C<0;[C@@H][C@@H]C>?{1}[C@@H]/C=C(CC)C/C(C=O)COC!C(CC)CC!?{4}CC!C=O"
            )
        )
        == "C=C/C=C\C<2;[C@@H]CCC=C>?{0}[C@@H]OC(C=O)C<0;[C@@H][C@@H]C>?{1}[C@@H]/C=C(CC)C/C(C=O)COC!C(CC)CC!?{4}CC!C=O"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CO<0;cccccc>?{2,3}<4;ccO[C@H][C@H]>?{3,4}<4;[C@H][C@H]OC=C>!!?{4,5}<1;occccc>?{1}c=O!?{2,3}<4;ccCCC>?{2}C=O!!!"
            )
        )
        == "CO<0;cccccc>?{2,3}<4;ccO[C@H][C@H]>?{3,4}<4;[C@H][C@H]OC=C>!!?{4,5}<1;occccc>?{1}c=O!?{2,3}<4;ccCCC>?{2}C=O!!!"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "OC<3;[C@H][C@@H][C@@H][C@H]O[C@@H]>?{0}[C@H]O!?{1}[C@@H]O!?{5}[C@@H]O![C@@H]O"
            )
        )
        == "OC<3;[C@H][C@@H][C@@H][C@H]O[C@@H]>?{0}[C@H]O!?{1}[C@@H]O!?{5}[C@@H]O![C@@H]O"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CO<0;cccccc>?{1}cO!?{3,4}<5;ccCO[C@@H][C@H]>?{2}C=O!?{4,5}<5;[C@H][C@@H][C@@H][C@H][C@@H]O>?{2}[C@@H]O!?{3}[C@H]O!?{4}[C@@H]CO!!!cO"
            )
        )
        == "CO<0;cccccc>?{1}cO!?{3,4}<5;ccCO[C@@H][C@H]>?{2}C=O!?{4,5}<5;[C@H][C@@H][C@@H][C@H][C@@H]O>?{2}[C@@H]O!?{3}[C@H]O!?{4}[C@@H]CO!!!cO"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "C=CCC[C@H]([C@H]C/C=C(CC)C\CCOC(CC)C=O)[C@H]C(C=C)CC"
            )
        )
        == "C=CCC[C@H]([C@H]C/C=C(CC)C\CCOC(CC)C=O)[C@H]C(C=C)CC"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CC(CC)C<0;[C@][C@H]C>?{0,1}<0;[C@]CC[C@H][C@H]>?{2}C=O!?{3}[C@H]C!!"
            )
        )
        == "CC(CC)C<0;[C@][C@H]C>?{0,1}<0;[C@]CC[C@H][C@H]>?{2}C=O!?{3}[C@H]C!!"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "C<5;ncnccc>?{4}cC<2;scc[n+]c>?{1}cCCO!cC!?{5}cN!"
            )
        )
        == "C<5;ncnccc>?{4}cC<2;scc[n+]c>?{1}cCCO!cC!?{5}cN!"
    )


# Ring ordering
def test_ring_ordering():
    assert (
        translate_to_own("F12CC1F1CC21", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        translate_to_own("F12CC1F1CC12", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        translate_to_own("F21CC1F1CC21", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        translate_to_own("F21CC1F1CC12", False)
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )


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
        translate_to_own("N#Cc3c(Oc1cccc(N)c1)ccc4nc(NC(=O)C2CC2)sc34", False)
        == "N#C<0;cccccc>?{1}cO<0;cccccc>?{4}cN!!?{4,5}<3;sccnc>?{4}cNC(C=O)C<0;CCC>!!"
    )
    assert (
        translate_to_own(
            "C/C=C/CC(C)C(O)C1C(=O)NC(CC)C(=O)N(C)CC(=O)N(C)C(CC(C)C)C(=O)NC(C(C)C)C(=O)N(C)C(CC(C)C)C(=O)NC(C)C(=O)NC(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(C(C)C)C(=O)N1C",
            False,
        )
        == "C/C=C/CC(CC)CC(CO)C<2;NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCNCC>?{1}CCC!?{2}C=O!?{3}NC!?{5}C=O!?{6}NC!?{7}CCC(CC)CC!?{8}C=O!?{%10}CC(CC)CC!?{%11}C=O!?{%12}NC!?{%13}CCC(CC)CC!?{%14}C=O!?{%16}CC!?{%17}C=O!?{%19}CC!?{%20}C=O!?{%21}NC!?{%22}CCC(CC)CC!?{%23}C=O!?{%24}NC!?{%25}CCC(CC)CC!?{%26}C=O!?{%27}NC!?{%28}CC(CC)CC!?{%29}C=O!?{%32}C=O!NC"
    )
    assert (
        translate_to_own(
            "CC[C@H]3OC(=O)[C@H](C)[C@@H](O[C@H]1C[C@@](C)(OC)[C@@H](O)[C@H](C)O1)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@H](N(C)C)[C@H]2O)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]3(C)O",
            False,
        )
        == "CC<13;[C@][C@H]OC[C@H][C@@H][C@H][C@@H][C@]C[C@@H]C[C@H][C@@H]>?{3}C=O!?{4}[C@H]C!?{5}[C@@H]O<4;[C@H]O[C@H]C[C@@][C@@H]>?{0}[C@H]C!?{4}[C@@]C!?{4}[C@@]OC!?{5}[C@@H]O!!?{6}[C@H]C!?{7}[C@@H]O<4;[C@H][C@H][C@@H]O[C@H]C>?{0}[C@H]N(NC)NC!?{4}[C@H]C![C@H]O!?{8}[C@]C!?{8}[C@]O!?{%10}[C@@H]C!?{%11}C=O!?{%12}[C@H]C!?{%13}[C@@H]O![C@]([C@]C)[C@]O"
    )
    assert (
        translate_to_own(
            "C#C[C@]4(O)CC[C@H]3[C@@H]2CCc1cc(O)ccc1[C@H]2CC[C@@]34C", False
        )
        == "C#C<0;[C@]CC[C@H][C@@]>?{0}[C@]O!?{3,4}<0;[C@H][C@@H][C@H]CC[C@@]>?{1,2}<3;cc[C@H][C@@H]CC>?{0,1}<0;cccccc>?{2}cO!!!![C@@]C"
    )
    assert (
        translate_to_own("N2CCCCCCCCC1CCCC1CCCCC2", False)
        == "<0;NCCCCCCCCCCCCCCC>?{9,%10}<0;CCCCC>!"
    )


def test_other_tokenisation():
    assert tokenise_our_representation("<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!") == [
        "<0;",
        "F",
        "C",
        "F",
        "C",
        ">",
        "?{",
        "0",
        ",",
        "1",
        "}",
        "<0;",
        "F",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "2",
        ",",
        "3",
        "}",
        "<0;",
        "F",
        "C",
        "C",
        ">",
        "!",
    ]
    assert tokenise_our_representation("F(<2;[Cl]OCF>CC)[Cl]F") == [
        "F",
        "(",
        "<2;",
        "[Cl]",
        "O",
        "C",
        "F",
        ">",
        "C",
        "C",
        ")",
        "[Cl]",
        "F",
    ]
    assert tokenise_our_representation(
        "N#C<0;cccccc>?{1}cO<0;cccccc>?{4}cN!!?{4,5}<3;sccnc>?{4}cNC(C=O)C<0;CCC>!!"
    ) == [
        "N",
        "#",
        "C",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "1",
        "}",
        "c",
        "O",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "4",
        "}",
        "c",
        "N",
        "!",
        "!",
        "?{",
        "4",
        ",",
        "5",
        "}",
        "<3;",
        "s",
        "c",
        "c",
        "n",
        "c",
        ">",
        "?{",
        "4",
        "}",
        "c",
        "N",
        "C",
        "(",
        "C",
        "=",
        "O",
        ")",
        "C",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "!",
    ]
    assert tokenise_our_representation(
        "C/C=C/CC(CC)CC(CO)C<2;NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCNCC>?{1}CCC!?{2}C=O!?{3}NC!?{5}C=O!?{6}NC!?{7}CCC(CC)CC!?{8}C=O!?{%10}CC(CC)CC!?{%11}C=O!?{%12}NC!?{%13}CCC(CC)CC!?{%14}C=O!?{%16}CC!?{%17}C=O!?{%19}CC!?{%20}C=O!?{%21}NC!?{%22}CCC(CC)CC!?{%23}C=O!?{%24}NC!?{%25}CCC(CC)CC!?{%26}C=O!?{%27}NC!?{%28}CC(CC)CC!?{%29}C=O!?{%32}C=O!NC"
    ) == [
        "C",
        "/",
        "C",
        "=",
        "C",
        "/",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "(",
        "C",
        "O",
        ")",
        "C",
        "<2;",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        "N",
        "C",
        "C",
        ">",
        "?{",
        "1",
        "}",
        "C",
        "C",
        "C",
        "!",
        "?{",
        "2",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "3",
        "}",
        "N",
        "C",
        "!",
        "?{",
        "5",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "6",
        "}",
        "N",
        "C",
        "!",
        "?{",
        "7",
        "}",
        "C",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "8",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%10",
        "}",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "%11",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%12",
        "}",
        "N",
        "C",
        "!",
        "?{",
        "%13",
        "}",
        "C",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "%14",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%16",
        "}",
        "C",
        "C",
        "!",
        "?{",
        "%17",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%19",
        "}",
        "C",
        "C",
        "!",
        "?{",
        "%20",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%21",
        "}",
        "N",
        "C",
        "!",
        "?{",
        "%22",
        "}",
        "C",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "%23",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%24",
        "}",
        "N",
        "C",
        "!",
        "?{",
        "%25",
        "}",
        "C",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "%26",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%27",
        "}",
        "N",
        "C",
        "!",
        "?{",
        "%28",
        "}",
        "C",
        "C",
        "(",
        "C",
        "C",
        ")",
        "C",
        "C",
        "!",
        "?{",
        "%29",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%32",
        "}",
        "C",
        "=",
        "O",
        "!",
        "N",
        "C",
    ]
    assert tokenise_our_representation(
        "CC<13;[C@][C@H]OC[C@H][C@@H][C@H][C@@H][C@]C[C@@H]C[C@H][C@@H]>?{3}C=O!?{4}[C@H]C!?{5}[C@@H]O<4;[C@H]O[C@H]C[C@@][C@@H]>?{0}[C@H]C!?{4}[C@@]C!?{4}[C@@]OC!?{5}[C@@H]O!!?{6}[C@H]C!?{7}[C@@H]O<4;[C@H][C@H][C@@H]O[C@H]C>?{0}[C@H]N(NC)NC!?{4}[C@H]C![C@H]O!?{8}[C@]C!?{8}[C@]O!?{%10}[C@@H]C!?{%11}C=O!?{%12}[C@H]C!?{%13}[C@@H]O![C@]([C@]C)[C@]O"
    ) == [
        "C",
        "C",
        "<13;",
        "[C@]",
        "[C@H]",
        "O",
        "C",
        "[C@H]",
        "[C@@H]",
        "[C@H]",
        "[C@@H]",
        "[C@]",
        "C",
        "[C@@H]",
        "C",
        "[C@H]",
        "[C@@H]",
        ">",
        "?{",
        "3",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "4",
        "}",
        "[C@H]",
        "C",
        "!",
        "?{",
        "5",
        "}",
        "[C@@H]",
        "O",
        "<4;",
        "[C@H]",
        "O",
        "[C@H]",
        "C",
        "[C@@]",
        "[C@@H]",
        ">",
        "?{",
        "0",
        "}",
        "[C@H]",
        "C",
        "!",
        "?{",
        "4",
        "}",
        "[C@@]",
        "C",
        "!",
        "?{",
        "4",
        "}",
        "[C@@]",
        "O",
        "C",
        "!",
        "?{",
        "5",
        "}",
        "[C@@H]",
        "O",
        "!",
        "!",
        "?{",
        "6",
        "}",
        "[C@H]",
        "C",
        "!",
        "?{",
        "7",
        "}",
        "[C@@H]",
        "O",
        "<4;",
        "[C@H]",
        "[C@H]",
        "[C@@H]",
        "O",
        "[C@H]",
        "C",
        ">",
        "?{",
        "0",
        "}",
        "[C@H]",
        "N",
        "(",
        "N",
        "C",
        ")",
        "N",
        "C",
        "!",
        "?{",
        "4",
        "}",
        "[C@H]",
        "C",
        "!",
        "[C@H]",
        "O",
        "!",
        "?{",
        "8",
        "}",
        "[C@]",
        "C",
        "!",
        "?{",
        "8",
        "}",
        "[C@]",
        "O",
        "!",
        "?{",
        "%10",
        "}",
        "[C@@H]",
        "C",
        "!",
        "?{",
        "%11",
        "}",
        "C",
        "=",
        "O",
        "!",
        "?{",
        "%12",
        "}",
        "[C@H]",
        "C",
        "!",
        "?{",
        "%13",
        "}",
        "[C@@H]",
        "O",
        "!",
        "[C@]",
        "(",
        "[C@]",
        "C",
        ")",
        "[C@]",
        "O",
    ]
    assert tokenise_our_representation(
        "C#C<0;[C@]CC[C@H][C@@]>?{0}[C@]O!?{3,4}<0;[C@H][C@@H][C@H]CC[C@@]>?{1,2}<3;cc[C@H][C@@H]CC>?{0,1}<0;cccccc>?{2}cO!!!![C@@]C"
    ) == [
        "C",
        "#",
        "C",
        "<0;",
        "[C@]",
        "C",
        "C",
        "[C@H]",
        "[C@@]",
        ">",
        "?{",
        "0",
        "}",
        "[C@]",
        "O",
        "!",
        "?{",
        "3",
        ",",
        "4",
        "}",
        "<0;",
        "[C@H]",
        "[C@@H]",
        "[C@H]",
        "C",
        "C",
        "[C@@]",
        ">",
        "?{",
        "1",
        ",",
        "2",
        "}",
        "<3;",
        "c",
        "c",
        "[C@H]",
        "[C@@H]",
        "C",
        "C",
        ">",
        "?{",
        "0",
        ",",
        "1",
        "}",
        "<0;",
        "c",
        "c",
        "c",
        "c",
        "c",
        "c",
        ">",
        "?{",
        "2",
        "}",
        "c",
        "O",
        "!",
        "!",
        "!",
        "!",
        "[C@@]",
        "C",
    ]
    assert tokenise_our_representation("<0;NCCCCCCCCCCCCCCC>?{9,%10}<0;CCCCC>!") == [
        "<0;",
        "N",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        ">",
        "?{",
        "9",
        ",",
        "%10",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        "C",
        "C",
        ">",
        "!",
    ]


def test_other_tokenisation_no_drop():
    assert (
        "".join(tokenise_our_representation("<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"))
        == "<0;FCFC>?{0,1}<0;FCC>!?{2,3}<0;FCC>!"
    )
    assert (
        "".join(tokenise_our_representation("F(<2;[Cl]OCF>CC)[Cl]F"))
        == "F(<2;[Cl]OCF>CC)[Cl]F"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "N#C<0;cccccc>?{1}cO<0;cccccc>?{4}cN!!?{4,5}<3;sccnc>?{4}cNC(C=O)C<0;CCC>!!"
            )
        )
        == "N#C<0;cccccc>?{1}cO<0;cccccc>?{4}cN!!?{4,5}<3;sccnc>?{4}cNC(C=O)C<0;CCC>!!"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "C/C=C/CC(CC)CC(CO)C<2;NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCNCC>?{1}CCC!?{2}C=O!?{3}NC!?{5}C=O!?{6}NC!?{7}CCC(CC)CC!?{8}C=O!?{%10}CC(CC)CC!?{%11}C=O!?{%12}NC!?{%13}CCC(CC)CC!?{%14}C=O!?{%16}CC!?{%17}C=O!?{%19}CC!?{%20}C=O!?{%21}NC!?{%22}CCC(CC)CC!?{%23}C=O!?{%24}NC!?{%25}CCC(CC)CC!?{%26}C=O!?{%27}NC!?{%28}CC(CC)CC!?{%29}C=O!?{%32}C=O!NC"
            )
        )
        == "C/C=C/CC(CC)CC(CO)C<2;NCCNCCNCCNCCNCCNCCNCCNCCNCCNCCNCC>?{1}CCC!?{2}C=O!?{3}NC!?{5}C=O!?{6}NC!?{7}CCC(CC)CC!?{8}C=O!?{%10}CC(CC)CC!?{%11}C=O!?{%12}NC!?{%13}CCC(CC)CC!?{%14}C=O!?{%16}CC!?{%17}C=O!?{%19}CC!?{%20}C=O!?{%21}NC!?{%22}CCC(CC)CC!?{%23}C=O!?{%24}NC!?{%25}CCC(CC)CC!?{%26}C=O!?{%27}NC!?{%28}CC(CC)CC!?{%29}C=O!?{%32}C=O!NC"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "CC<13;[C@][C@H]OC[C@H][C@@H][C@H][C@@H][C@]C[C@@H]C[C@H][C@@H]>?{3}C=O!?{4}[C@H]C!?{5}[C@@H]O<4;[C@H]O[C@H]C[C@@][C@@H]>?{0}[C@H]C!?{4}[C@@]C!?{4}[C@@]OC!?{5}[C@@H]O!!?{6}[C@H]C!?{7}[C@@H]O<4;[C@H][C@H][C@@H]O[C@H]C>?{0}[C@H]N(NC)NC!?{4}[C@H]C![C@H]O!?{8}[C@]C!?{8}[C@]O!?{%10}[C@@H]C!?{%11}C=O!?{%12}[C@H]C!?{%13}[C@@H]O![C@]([C@]C)[C@]O"
            )
        )
        == "CC<13;[C@][C@H]OC[C@H][C@@H][C@H][C@@H][C@]C[C@@H]C[C@H][C@@H]>?{3}C=O!?{4}[C@H]C!?{5}[C@@H]O<4;[C@H]O[C@H]C[C@@][C@@H]>?{0}[C@H]C!?{4}[C@@]C!?{4}[C@@]OC!?{5}[C@@H]O!!?{6}[C@H]C!?{7}[C@@H]O<4;[C@H][C@H][C@@H]O[C@H]C>?{0}[C@H]N(NC)NC!?{4}[C@H]C![C@H]O!?{8}[C@]C!?{8}[C@]O!?{%10}[C@@H]C!?{%11}C=O!?{%12}[C@H]C!?{%13}[C@@H]O![C@]([C@]C)[C@]O"
    )
    assert (
        "".join(
            tokenise_our_representation(
                "C#C<0;[C@]CC[C@H][C@@]>?{0}[C@]O!?{3,4}<0;[C@H][C@@H][C@H]CC[C@@]>?{1,2}<3;cc[C@H][C@@H]CC>?{0,1}<0;cccccc>?{2}cO!!!![C@@]C"
            )
        )
        == "C#C<0;[C@]CC[C@H][C@@]>?{0}[C@]O!?{3,4}<0;[C@H][C@@H][C@H]CC[C@@]>?{1,2}<3;cc[C@H][C@@H]CC>?{0,1}<0;cccccc>?{2}cO!!!![C@@]C"
    )
    assert (
        "".join(tokenise_our_representation("<0;NCCCCCCCCCCCCCCC>?{9,%10}<0;CCCCC>!"))
        == "<0;NCCCCCCCCCCCCCCC>?{9,%10}<0;CCCCC>!"
    )


def test_edgecases():
    assert translate_to_own("CCC1CC(C)C1C") == "CC<0;CCCC>?{2}CC!CC"
    assert translate_to_own("CCC1C(C)C(C)C1F") == "CC<0;CCCC>?{1}CC!?{2}CC!CF"
    assert (
        translate_to_own(
            "C%17CCCC1CC1C2CC2C3CC3C4CC4C5CC5C6CC6C7CC7C8CC8C9CC9C%10CC%10C%11CC%11C%12CC%12C%13CC%13C%14CC%14C%15CC%15C%16CC%16CC%17"
        )
        == "<0;CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC>?{4,5}<0;CCC>!?{6,7}<0;CCC>!?{8,9}<0;CCC>!?{%10,%11}<0;CCC>!?{%12,%13}<0;CCC>!?{%14,%15}<0;CCC>!?{%16,%17}<0;CCC>!?{%18,%19}<0;CCC>!?{%20,%21}<0;CCC>!?{%22,%23}<0;CCC>!?{%24,%25}<0;CCC>!?{%26,%27}<0;CCC>!?{%28,%29}<0;CCC>!?{%30,%31}<0;CCC>!?{%32,%33}<0;CCC>!?{%34,%35}<0;CCC>!"
    )
    assert tokenise_our_representation(
        translate_to_own(
            "C%17CCCC1CC1C2CC2C3CC3C4CC4C5CC5C6CC6C7CC7C8CC8C9CC9C%10CC%10C%11CC%11C%12CC%12C%13CC%13C%14CC%14C%15CC%15C%16CC%16CC%17"
        )
    ) == [
        "<0;",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        ">",
        "?{",
        "4",
        ",",
        "5",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "6",
        ",",
        "7",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "8",
        ",",
        "9",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%10",
        ",",
        "%11",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%12",
        ",",
        "%13",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%14",
        ",",
        "%15",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%16",
        ",",
        "%17",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%18",
        ",",
        "%19",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%20",
        ",",
        "%21",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%22",
        ",",
        "%23",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%24",
        ",",
        "%25",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%26",
        ",",
        "%27",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%28",
        ",",
        "%29",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%30",
        ",",
        "%31",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%32",
        ",",
        "%33",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
        "?{",
        "%34",
        ",",
        "%35",
        "}",
        "<0;",
        "C",
        "C",
        "C",
        ">",
        "!",
    ]


def test_polarity():
    assert (
        translate_to_own("C2CC(C1[CH2+2]C[F+]1)C2") == "<0;CCCC>?{2}C<3;[F+]C[CH2+2]C>!"
    )
    assert tokenise_our_representation("<0;CCCC>?{2}C<3;[F+]C[CH2+2]C>!") == [
        "<0;",
        "C",
        "C",
        "C",
        "C",
        ">",
        "?{",
        "2",
        "}",
        "C",
        "<3;",
        "[F+]",
        "C",
        "[CH2+2]",
        "C",
        ">",
        "!",
    ]
    assert (
        "".join(tokenise_our_representation("<0;CCCC>?{2}C<3;[F+]C[CH2+2]C>!"))
        == "<0;CCCC>?{2}C<3;[F+]C[CH2+2]C>!"
    )


def test_atom_ids():
    assert (
        translate_to_own("C2[C:%12]C(C1[CH2+2:2]C[F+:1]1)C2")
        == "<1;[C:%12]CCC>?{1}C<3;[F+:1]C[CH2+2:2]C>!"
    )
    assert tokenise_our_representation("<1;[C:%12]CCC>?{1}C<3;[F+:1]C[CH2+2:2]C>!") == [
        "<1;",
        "[C:%12]",
        "C",
        "C",
        "C",
        ">",
        "?{",
        "1",
        "}",
        "C",
        "<3;",
        "[F+:1]",
        "C",
        "[CH2+2:2]",
        "C",
        ">",
        "!",
    ]
    assert (
        "".join(
            tokenise_our_representation("<1;[C:%12]CCC>?{1}C<3;[F+:1]C[CH2+2:2]C>!")
        )
        == "<1;[C:%12]CCC>?{1}C<3;[F+:1]C[CH2+2:2]C>!"
    )


if __name__ == "__main__":
    test_ring_ordering()
    test_wikipedia_examples()
    test_ring_in_branch()
    test_hard_smiles()
    test_ring_ordering_and_perc()
    test_wikipedia_tokenisation()
    test_wikipedia_tokenisation_no_drop()
    test_other_tokenisation()
    test_other_tokenisation_no_drop()
