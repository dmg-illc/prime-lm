from glob import glob
import os


def capital(s):
    return s[0].upper() + s[1:]


def init_corpora(data_dir, tokenizer, add_eos=False):
    """ Processes the corpora in a way that is compatible with the pipeline and the current tokenizer. """
    primed_corpora = {}

    corpus_attributions = {}

    glob_str = os.path.join(data_dir, "*.csv")
    sep = ","
    bos = tokenizer.bos_token or tokenizer.cls_token
    skip_first_line = True
    sen_sep = ""

    if "roberta" in tokenizer.name_or_path:
        sen_sep = tokenizer.eos_token + tokenizer.eos_token
    elif "bert" in tokenizer.name_or_path:
        sen_sep = tokenizer.sep_token

    COLUMNS = {
        "x_px": "prime_x_start_idx",
        "x_py": "prime_y_start_idx",
        "y_px": "prime_x_start_idx",
        "y_py": "prime_y_start_idx",
    }

    if not os.path.isdir("new_corpora"):
        os.mkdir("new_corpora")

    for corpus_path in glob(glob_str):
        with open(corpus_path) as f:
            lines = [l.strip().split(sep) for l in f]

        new_lines = [
            ",".join(
                list(COLUMNS.keys())
                + ["unprimed_start_idx", "prime_x_start_idx", "prime_y_start_idx"]
            )
        ]

        for line in lines[int(skip_first_line) :]:
            # The final replace here hard codes captialization in the second sentence for recency & cumulativity
            # NB ==> Make sure that if the template changes this is updated accordingly here.
            prime_x: str = capital(
                line[0]
                .strip()
                .replace(" .", ".")
                .replace(". the", ". The")
                .replace(". a", ". A")
            )
            prime_y: str = capital(
                line[1]
                .strip()
                .replace(" .", ".")
                .replace(". the", ". The")
                .replace(". a", ". A")
            )

            prime_x = bos + prime_x + sen_sep
            prime_y = bos + prime_y + sen_sep

            x: str = capital(line[2].strip().replace(" .", "."))
            y: str = capital(line[3].strip().replace(" .", "."))

            if add_eos:
                x += tokenizer.eos_token
                y += tokenizer.eos_token

            new_line = []
            if "prime_x" in COLUMNS:
                new_line.append(prime_x)
            if "prime_y" in COLUMNS:
                new_line.append(prime_y)
            if "x" in COLUMNS:
                new_line.append(bos + x)
            if "y" in COLUMNS:
                new_line.append(bos + y)
            if "x_px" in COLUMNS:
                new_line.append(" ".join((prime_x, x)))
            if "x_py" in COLUMNS:
                new_line.append(" ".join((prime_y, x)))
            if "y_px" in COLUMNS:
                new_line.append(" ".join((prime_x, y)))
            if "y_py" in COLUMNS:
                new_line.append(" ".join((prime_y, y)))

            prime_x_start_idx = len(tokenizer.tokenize(prime_x))
            prime_y_start_idx = len(tokenizer.tokenize(prime_y))

            new_line.extend(
                [
                    "1",
                    str(prime_x_start_idx),
                    str(prime_y_start_idx),
                ]
            )

            new_lines.append(",".join(new_line))

        corpus_name = corpus_path.split("/")[-1].split(".")[0]
        new_corpus_name = f"{corpus_name}_processed.csv"
        new_corpus_path = os.path.join("new_corpora", new_corpus_name)

        with open(new_corpus_path, "w") as f:
            f.write("\n".join(new_lines))

        primed_corpora[corpus_name] = new_corpus_path

    return primed_corpora, COLUMNS
