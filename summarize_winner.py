import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def dedupe_per_abstract(entries):
    """Collapse repeated pmid sightings of the same abstract into one entry.

    The same PMID is often re-fetched independently in every round/iteration it
    appears in. Group by pmid and merge evidence sentences (deduped, order preserved).
    """
    grouped = defaultdict(list)
    for e in entries:
        grouped[e["pmid"]].append(e)

    deduped = []
    for pmid, group in grouped.items():
        evidence = []
        seen = set()
        for e in group:
            for ev in e.get("evidence") or []:
                if ev not in seen:
                    seen.add(ev)
                    evidence.append(ev)
        deduped.append({"pmid": pmid, "label": "support", "evidence": evidence})
    tallies = {"support": len(deduped)}
    return deduped, tallies


def find_iteration_files(pair_results_dir):
    """Return [(iteration_num, json_path), ...] for a pair's results dir, iterated or not."""
    iter_dirs = sorted(
        d for d in os.listdir(pair_results_dir)
        if d.startswith("iteration_") and os.path.isdir(os.path.join(pair_results_dir, d))
    )
    files = []
    if iter_dirs:
        for d in iter_dirs:
            matches = glob.glob(os.path.join(pair_results_dir, d, "*_km_with_gpt_direct_comp.json"))
            if matches:
                files.append((int(d.split("_", 1)[1]), matches[0]))
    else:
        matches = sorted(glob.glob(os.path.join(pair_results_dir, "*_km_with_gpt_direct_comp.json")))
        files = [(i, m) for i, m in enumerate(matches, start=1)]
    return files


def load_result_entry(json_path):
    data = json.load(open(json_path))
    rec = data[0] if isinstance(data, list) else data
    hc = rec["Hypothesis_Comparison"]
    result = hc["Result"]
    return result[0] if isinstance(result, list) else result


def summarize_winner(tournament_dir, winner=None):
    history = json.load(open(os.path.join(tournament_dir, "bracket_history.json")))

    final_terms = history["final_terms"]
    if winner is None:
        if len(final_terms) != 1:
            raise SystemExit(
                f"Tournament ended with {len(final_terms)} co-winners ({final_terms}); "
                "pass -winner to pick which one to summarize."
            )
        winner = final_terms[0]
    elif winner not in final_terms:
        raise SystemExit(f"'{winner}' is not among the final term(s) {final_terms}")

    raw_per_abstract = []
    scores = []
    round_rationales = []

    for round_result in history["rounds"]:
        for pair in round_result["pairs"]:
            if pair["status"] != "ok":
                continue
            if winner == pair["term1"]:
                position = "H1"
            elif winner == pair["term2"]:
                position = "H2"
            else:
                continue

            pair_dir = os.path.join(
                tournament_dir, f"round_{round_result['round']}", "output", pair["pair_id"], "results"
            )
            if not os.path.isdir(pair_dir):
                continue

            # "supports_H2" (when winner=H1) isn't evidence against the winner's gene --
            # it's just evidence for the other gene. So only tally the winner's own
            # supporting abstracts; there's no meaningful "refute" category here.
            support_label = "supports_H1" if position == "H1" else "supports_H2"
            pair_rationale_sentences = []

            for _iteration_num, json_path in find_iteration_files(pair_dir):
                entry = load_result_entry(json_path)

                raw_score = float(entry["score"])
                scores.append(raw_score if position == "H1" else (100 - raw_score))

                for ab in entry.get("per_abstract", []):
                    if ab.get("label") == support_label:
                        raw_per_abstract.append({"pmid": ab.get("pmid"), "evidence": ab.get("evidence")})

                pair_rationale_sentences.extend(entry.get("score_rationale", []))

            opponent = pair["term2"] if position == "H1" else pair["term1"]
            round_rationales.append({
                "round": round_result["round"],
                "pair_id": pair["pair_id"],
                "opponent": opponent,
                "winner_position": position,
                # Raw sentences straight from each iteration's score_rationale, deduped.
                # This is a placeholder for overall_rationale -- replace each round's
                # raw list with a genuine 1-2 sentence synthesis before treating this
                # file as final.
                "raw_rationale_sentences": list(dict.fromkeys(pair_rationale_sentences)),
            })

    overall_score = statistics.mean(scores) if scores else None
    per_abstract, tallies = dedupe_per_abstract(raw_per_abstract)

    return {
        "Winner": winner,
        "Result": {
            "per_abstract": per_abstract,
            "tallies": tallies,
            "overall_rationale": [r["raw_rationale_sentences"] for r in round_rationales],
            "overall_score": overall_score,
            "scores": scores,
        },
    }, round_rationales


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a tournament winner's evidence across every round it participated in."
    )
    parser.add_argument("tournament_dir")
    parser.add_argument("-winner", default=None, help="Which term to summarize if the tournament ended in a tie")
    parser.add_argument("-out", default="winner_summary.json")
    args = parser.parse_args()

    result, round_rationales = summarize_winner(args.tournament_dir, args.winner)

    out_path = os.path.join(args.tournament_dir, args.out)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {out_path}")

    raw_path = os.path.join(args.tournament_dir, "winner_round_rationales_raw.json")
    with open(raw_path, "w") as f:
        json.dump(round_rationales, f, indent=2)
    print(f"Wrote {raw_path} (raw per-round rationale sentences, for synthesizing overall_rationale)")


if __name__ == "__main__":
    main()
