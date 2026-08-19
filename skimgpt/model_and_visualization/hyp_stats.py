import cmdlogtime
import os
import sys
from pathlib import Path
from scipy.stats import chi2_contingency as chisq
from scipy.stats import ks_2samp as ks2
from scipy.stats import mannwhitneyu as mw
import numpy as np
from collections import OrderedDict as OD
from scipy.stats import binomtest
from scipy.stats import multinomial
from scipy.stats import fisher_exact
from statsmodels.stats.proportion import tost_proportions_2indep as tost
from statsmodels.stats.proportion import proportions_ztest as z_prop_test

COMMAND_LINE_DEF_FILE = str(Path(__file__).parent / "hyp_stats_commandline.txt")

def main():
    (start_time_secs, pretty_start_time, my_args, addl_logfile) = cmdlogtime.begin(COMMAND_LINE_DEF_FILE)
    out_file_km = os.path.join(my_args["out_dir"], "km_hyp_stats.txt")
    out_file_skim = os.path.join(my_args["out_dir"], "skim_hyp_stats.txt")
    out_file_km_kept = os.path.join(my_args["out_dir"], "km_kept.txt")
    out_file_skim_kept = os.path.join(my_args["out_dir"], "skim_kept.txt")
    skip_skim = my_args["skip_skim_stats"]
    date_hold = ""
    lines_hold = []
    in_terms = my_args["terms"].strip().split(",")
    in_files = [os.path.join(my_args["in_dir"], "km_hyp.txt")]
    if not skip_skim:
        in_files.append(os.path.join(my_args["in_dir"], "skim_hyp.txt"))
    file_ctr = 0
    for in_file in in_files:  # read KM and SKiM  infiles and do stats
        file_ctr = file_ctr + 1
        line_ctr = 0
        if file_ctr == 1:
            out_file = out_file_km
            out_file_kept = out_file_km_kept
        else:
            out_file = out_file_skim
            out_file_kept = out_file_skim_kept
        lines_prior_year = []
        with open(out_file, "w") as out_f, open(in_file, "r") as in_f, open(out_file_kept, "w") as out_f_kept:
            print("Processing file: ", in_file)
            lines = in_f.readlines()
            for line in lines:
                line_ctr = line_ctr + 1
                if line_ctr == 1:
                    out_f_kept.write(line)
                    out_f.write("Year\tStatType\tTerms\tStatistic\tP-Value\tAdditional Info\n")
                    continue  # skip header line
                curr_line = line.strip().split("\t")
                curr_date = curr_line[0]
                if line_ctr == 2:
                    date_hold = curr_date
                if curr_date != date_hold:  # date has changed, so process the prior line(s) and calc the stats
                    (terms, stats, p_vals, kept_lines, addl_info) = parse_lines_calc_statistic_and_pval(
                        lines_hold,
                        file_ctr,
                        lines_prior_year,
                        my_args["fet_ab"],
                        my_args["fet_bc"],
                        in_terms,
                        out_f_kept,
                    )
                    for key in stats:
                        if ":" in key:
                            stat_type, terms = key.split(":")  # NOTE that stat_type or terms should not contain a ":"
                        else:
                            stat_type = key
                            terms = ""
                        out_f.write(
                            str(date_hold) + "\t" + str(stat_type) + "\t" + str(terms) + "\t" + str(stats[key]) + "\t" + str(p_vals[key]) + "\t" + str(addl_info[key]) + "\n"
                        )
                    lines_prior_year = kept_lines
                    lines_hold = []
                lines_hold.append(curr_line)
                date_hold = curr_date
            # Capture last one here
            (terms, stats, p_vals, kept_lines, addl_info) = parse_lines_calc_statistic_and_pval(
                lines_hold,
                file_ctr,
                lines_prior_year,
                my_args["fet_ab"],
                my_args["fet_bc"],
                in_terms,
                out_f_kept,
            )
            for key in stats:
                if ":" in key:
                    stat_type, terms = key.split(":")  # NOTE that stat_type or terms should not contain a ":"
                else:
                    stat_type = key
                    terms = ""
                out_f.write(str(date_hold) + "\t" + str(stat_type) + "\t" + str(terms) + "\t" + str(stats[key]) + "\t" + str(p_vals[key]) + "\t" + str(addl_info[key]) + "\n")
        lines_hold = []

    cmdlogtime.end(addl_logfile, start_time_secs)

# ---------------- FUNCTIONS --------------------
def parse_lines_calc_statistic_and_pval(
    lines,
    file_ctr,
    lines_prior_year,
    fet_ab,
    fet_bc,
    in_terms,
    out_f_kept,
):
    (sorted_lines, sorted_prior_year_lines, kept_lines, unique_terms, sort_index) = filter_and_sort_lines(file_ctr, lines, fet_ab, fet_bc, lines_prior_year, in_terms, out_f_kept)
    terms, stats, p_vals, addl_info = check_for_no_or_only1_line(sorted_lines, unique_terms, sort_index)
    if "NOLINES" in stats or "ONLY1" in stats:
        return terms, stats, p_vals, kept_lines, addl_info
    d = initialize_dicts()
    terms = []
    if file_ctr == 1:  # KM case
        d, terms = load_km_dicts(d, sorted_lines, sorted_prior_year_lines, terms)
        terms = reorder_terms(terms, in_terms)
        stats, p_vals, addl_info = calc_rat_of_rat(stats, p_vals, terms, d, "KM", addl_info)
        stats, p_vals, addl_info = calc_ci_of_oddsratio(stats, p_vals, terms, d, "KM", addl_info)
        stats, p_vals, addl_info = calc_chi_square(stats, p_vals, terms, d, "KM", addl_info)
        return terms, stats, p_vals, kept_lines, addl_info
    else:  # SKIM Case
        d, terms = load_skim_dicts(d, sorted_lines, sorted_prior_year_lines, terms)
        terms = reorder_terms(terms, in_terms)
        stats, p_vals, addl_info = calc_rat_of_rat(stats, p_vals, terms, d, "SKiM", addl_info)
        stats, p_vals, addl_info = calc_ci_of_oddsratio(stats, p_vals, terms, d, "SKiM", addl_info)
        stats, p_vals, addl_info = calc_binomtest(stats, p_vals, terms, d, addl_info)
        stats, p_vals, addl_info = calc_chi_square(stats, p_vals, terms, d, "SKiM", addl_info)
        return terms, stats, p_vals, kept_lines, addl_info

def initialize_dicts():
    d = OD()  # big orderded dictionary of ordered dictionaries
    d["ratios"] = OD()
    d["pred_scores"] = OD()
    d["pred_scores_sum"] = OD()
    d["fet_pvals"] = OD()
    d["ratios_sum"] = OD()  # dictionary, key is term, and it holds a number,  used for SKiM case
    d["km_a_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["km_b_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["skim_a_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["skim_b_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["skim_ab_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["skim_b_terms"] = OD()  # dictionary, key is term, and it holds a list
    d["skim_c_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["skim_c_terms"] = OD()  # dictionary, key is term, and it holds a list
    d["tot_counts"] = OD()  # dictionary, key is term, and it holds a list (total counts)
    d["intersection_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["intersection_counts_sum"] = OD()  # dictionary, key is term, and it holds a number
    d["c_counts_sum"] = OD()  # dictionary, key is term, and it holds a number
    d["b_counts_sum"] = OD()  # dictionary, key is term, and it holds a number
    d["prior_year_a_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["prior_year_b_counts"] = OD()  # dictionary, key is term, and it holds a list
    d["prior_year_ab_counts"] = OD()  # dictionary, key is term, and it holds a list
    return d

def load_km_dicts(d, sorted_lines, sorted_prior_year_lines, terms):
    # date, a_term, a_count, b_term, b_count, ab_count, ab_pmids, ab_pred_score, ab_pval, ab_ratio, total_count
    for line in sorted_lines:
        term = line[3]
        if term not in d["prior_year_a_counts"]:
            d["prior_year_a_counts"][term] = []
            d["prior_year_a_counts"][term].append(0)
            d["prior_year_b_counts"][term] = []
            d["prior_year_b_counts"][term].append(0)
            d["prior_year_ab_counts"][term] = []
            d["prior_year_ab_counts"][term].append(0)
    for line2 in sorted_prior_year_lines:  # rms. just put a term count from prior year in for all key=terms .  as a baseline for the year
        term2 = line2[3]
        d["prior_year_a_counts"][term2][0] = float(line2[2])
        d["prior_year_b_counts"][term2][0] = float(line2[4])
        d["prior_year_ab_counts"][term2][0] = float(line2[5])

    for line in sorted_lines:
        term = line[3]
        if term not in d["ratios"]:
            # technically, these don't need to be lists, but I am making them lists, as they need to be lists for skim
            d["ratios"][term] = []
            d["km_a_counts"][term] = []
            d["km_b_counts"][term] = []
            d["tot_counts"][term] = []
            d["intersection_counts"][term] = []  # Note that intersection_counts in the KM case is the AB count.
            d["fet_pvals"][term] = []
            d["pred_scores"][term] = []  # Add this line to initialize pred_scores for each term
            terms.append(term)
        d["ratios"][term].append(float(line[9]))
        d["km_a_counts"][term].append(float(line[2]))  # not time-windowing
        d["km_b_counts"][term].append(float(line[4]))
        d["intersection_counts"][term].append(float(line[5]))
        d["pred_scores"][term].append(float(line[7]))
        d["fet_pvals"][term].append(float(line[8]))
        d["tot_counts"][term].append(float(line[10]))
    return d, terms


def load_skim_dicts(d, sorted_lines, sorted_prior_year_lines, terms):
    # Date	A_Term	A_Count	B_Term	B_Count	AB_Count	AB_PMIDS	AB_Pred_Score	AB_Pvalue	AB_Sort_Ratio
    # B_term	BC_Count	BC_PMIDS	BC_Pred_Score	BC_Pvalue	BC_sort_ratio	C_term	C_count	Total_count FET_BC Cutoff
    for line in sorted_lines:
        term = line[16]
        if term not in d["ratios"]:
            d["ratios"][term] = []
            d["skim_a_counts"][term] = []
            d["skim_b_counts"][term] = []
            d["skim_b_terms"][term] = []
            d["skim_ab_counts"][term] = []
            d["skim_c_counts"][term] = []
            d["skim_c_terms"][term] = []
            d["tot_counts"][term] = []
            d["intersection_counts"][term] = []   # Note that intersection_counts in the SKiM case is the BC count.
            d["ratios_sum"][term] = 0
            d["intersection_counts_sum"][term] = 0
            d["pred_scores"][term] = []
            d["pred_scores_sum"][term] = 0
            d["c_counts_sum"][term] = 0
            d["b_counts_sum"][term] = 0
            terms.append(term)
        d["ratios"][term].append(float(line[15]))
        d["ratios_sum"][term] = d["ratios_sum"][term] + float(line[15])
        d["intersection_counts"][term].append(float(line[11]))  # Note that intersection_counts in the SKiM case is the BC count.
        d["intersection_counts_sum"][term] = d["intersection_counts_sum"][term] + float(line[11])
        d["pred_scores"][term].append(float(line[13]))
        d["pred_scores_sum"][term] = d["pred_scores_sum"][term] + float(line[13])
        d["c_counts_sum"][term] = d["c_counts_sum"][term] + float(line[17])
        d["b_counts_sum"][term] = d["b_counts_sum"][term] + float(line[4])
        d["skim_a_counts"][term].append(float(line[2]))
        d["skim_b_counts"][term].append(float(line[4]))
        d["skim_ab_counts"][term].append(float(line[5]))
        d["skim_b_terms"][term].append(line[3])
        d["skim_c_terms"][term].append(line[16])
        d["skim_c_counts"][term].append(float(line[17]))
        d["tot_counts"][term].append(float(line[18]))
    return d, terms

def calc_binomtest(stats, p_vals, terms, d, addl_info):
    # try binomial test of proportions
    # compare the number of the first B terms vs the total # of B terms
    # but normalize for the number of C term counts.
    num_first_b_terms = len(d["ratios"][terms[0]])
    num_second_b_terms = round(len(d["ratios"][terms[1]]) * (d["skim_c_counts"][terms[0]][0] / d["skim_c_counts"][terms[1]][0]))
    tot_b_terms = num_first_b_terms + num_second_b_terms
    binom_gt = binomtest(num_first_b_terms, tot_b_terms, p=0.5, alternative="greater")
    binom_lt = binomtest(num_first_b_terms, tot_b_terms, p=0.5, alternative="less")
    key = "Binom. GT:" + terms[0] + "," + terms[1]
    stats[key] = binom_gt.statistic
    p_vals[key] = binom_gt.pvalue
    addl_info[key] = binom_gt
    key = "Binom. LT:" + terms[0] + "," + terms[1]
    stats[key] = binom_lt.statistic
    p_vals[key] = binom_lt.pvalue
    addl_info[key] = binom_lt
    # modify num_first_b_terms by the ratio of the average ratios across 1st and 2nd B terms
    # and compute a weighted binomial statistic
    average_first_ratios = np.average(d["ratios"][terms[0]])
    average_second_ratios = np.average(d["ratios"][terms[1]])
    mod_num_first_b_terms = round(num_first_b_terms * average_first_ratios / average_second_ratios)
    if (mod_num_first_b_terms == 0):  # don't want to round to zero, or will get error
        mod_num_first_b_terms = 1
    tot_mod_b_terms = mod_num_first_b_terms + num_second_b_terms
    binom_gt = binomtest(mod_num_first_b_terms, tot_mod_b_terms, p=0.5, alternative="greater")
    binom_lt = binomtest(mod_num_first_b_terms, tot_mod_b_terms, p=0.5, alternative="less")
    key = "Weighted Binom. GT:" + terms[0] + "," + terms[1]
    stats[key] = binom_gt.statistic
    p_vals[key] = binom_gt.pvalue
    addl_info[key] = binom_gt
    key = "Weighted Binom. LT:" + terms[0] + "," + terms[1]
    stats[key] = binom_lt.statistic
    p_vals[key] = binom_lt.pvalue
    addl_info[key] = binom_lt
    return stats, p_vals, addl_info

def calc_chi_square(stats, p_vals, terms, d, run_type, addl_info):
    # chi-square test.
    # FOR skim case:
    # chisq test of summed BC and B numbers.
    # chi-square test
    if run_type == "SKiM":
        # normalize intersection counts for term 1 to correct for different C counts between terms 1 &2
        normed_intersection_counts_sum_terms1 = d["intersection_counts_sum"][terms[1]] * (d["skim_c_counts"][terms[0]][0] / d["skim_c_counts"][terms[1]][0])
        # use b_counts sum, not c_counts_sum
        f_obs2 = [
            [d["intersection_counts_sum"][terms[0]], d["b_counts_sum"][terms[0]]],
            [normed_intersection_counts_sum_terms1, d["b_counts_sum"][terms[1]]],
        ]
        chisq_stat, chisq_p, chisq_dof, chisq_expected = chisq(f_obs2, correction=True)
        stats["chisq_b_summed"] = chisq_stat
        p_vals["chisq_b_summed"] = chisq_p
        addl_info["chisq_b_summed"] = ""
        f_obs = [
            [d["intersection_counts_sum"][terms[0]], d["c_counts_sum"][terms[0]]],
            [d["intersection_counts_sum"][terms[1]], d["c_counts_sum"][terms[1]]],
        ]
    else:  # KM
        f_obs = [
            [d["intersection_counts"][terms[0]][0], d["km_b_counts"][terms[0]][0]],
            [d["intersection_counts"][terms[1]][0], d["km_b_counts"][terms[1]][0]],
        ]
    chisq_stat, chisq_p, chisq_dof, chisq_expected = chisq(f_obs, correction=True)
    stats["chisq"] = chisq_stat
    p_vals["chisq"] = chisq_p
    addl_info["chisq"] = ""
    return stats, p_vals, addl_info

def calc_rat_of_rat(stats, p_vals, terms, d, run_type, addl_info):
    if run_type == "SKiM":
        key = "difference_in_sums_of_intersection:" + str([terms[0]][0]) + "," + str([terms[1]][0])
        # stat based on just the difference in the sums of the B-C intersection.
        # I will only normalize the statistic for display purposes.
        # The un-normalized values will be compared to permuted un-normalized values
        normed_obs_diff = (d["intersection_counts_sum"][terms[0]] - d["intersection_counts_sum"][terms[1]]) * (d["skim_c_counts"][terms[1]][0] / d["skim_c_counts"][terms[0]][0])
        obs_diff = d["intersection_counts_sum"][terms[0]] - d["intersection_counts_sum"][terms[1]]
        stats[key] = normed_obs_diff
        #  do a permutation test to see of the difference between the sums of the intersection counts are different
        # Calculate the observed difference in sums
        p_vals[key] = perm_test(d["intersection_counts"][terms[0]], d["intersection_counts"][terms[1]], "diff", obs_diff, num_permutations=10000)
        addl_info[key] = "Permutation method for difference of sums is probably ok."
        # Testing ratios of intersection sums
        # get bc and c total counts
        BC1 = d["intersection_counts_sum"][terms[0]]  # Number of hits for BC1
        C1_tot = d["skim_c_counts"][terms[0]][0]  # Total number of C1  #RMS. Maybe should multiply by number of B terms that intersect with C1
        BC2 = d["intersection_counts_sum"][terms[1]]  # Number of hits for BC2
        C2_tot = d["skim_c_counts"][terms[1]][0]  # Total number of C2   #RMS. Maybe should multiply by number of B terms that intersect with C2

        # z test of proportions.
        key = "ratio_in_sums_of_intersection_zProp:" + str([terms[0]][0]) + "," + str([terms[1]][0])
        z_prop_stat, p_value_prop = z_prop_test([BC1, BC2], [C1_tot, C2_tot])
        p_vals[key] = p_value_prop
        stats[key] = z_prop_stat
        addl_info[key] = "Reporting z proportions p-value.  stat:" + str(z_prop_stat)
    else:  # KM
        # get counts based on the AB hits and the B totals
        a = d["intersection_counts"][terms[0]][0]  # Number of ab hits for 1st term
        a_tot = d["km_b_counts"][terms[0]][0]  # Total number of b hits for 1st term
        b = d["intersection_counts"][terms[1]][0]  # Number of ab hits for 2nd term
        b_tot = d["km_b_counts"][terms[1]][0]  # Total number of b for 2nd term
        # Do z test of proportions using count values.
        key = "ratio_of_ratios_zprop:" + str([terms[0]][0]) + "," + str([terms[1]][0])
        z_prop_stat, p_value_prop = z_prop_test([a, b], [a_tot, b_tot])
        stats[key] = z_prop_stat
        p_vals[key] = p_value_prop
        addl_info[key] = "Reporting on z test of proportions."
    return stats, p_vals, addl_info

def perm_test(list1, list2, comp_type, obs_val, num_permutations=10000):
    # comp_type is either "rat" or "diff"
    # obs_val is the observed value of the ratio or difference
    if comp_type == "diff":
        obs_diff_pos = True
        if obs_val < 0:
            obs_diff_pos = False
    # Combine the datasets
    combined = list1 + list2
    # Perform permutation test
    n_permutations = 10000
    count = 0
    for _ in range(n_permutations):
        permuted = np.random.permutation(combined)
        rand_len = np.random.randint(len(combined) + 1)
        perm_list1 = permuted[:rand_len]
        perm_list2 = permuted[rand_len:]
        if comp_type == "diff":
            perm_diff = sum(perm_list1) - sum(perm_list2)
            if obs_diff_pos:
                if perm_diff >= obs_val:
                    count += 1
            else:
                if perm_diff <= obs_val:
                    count += 1
        else:  # comp_type == "rat"
            if sum(perm_list2) == 0:
                perm_ratio = obs_val
            else:
                perm_ratio = sum(perm_list1) / sum(perm_list2)
            if perm_ratio >= obs_val:
                count += 1
    # Calculate p-value
    perm_pval = count / n_permutations
    return perm_pval

def build_standard_contingency_table(a, b, ab, tot):
    # Construct the contingency table
    #                     b hits        not b_hits
    #                   --------     --------
    #      a hits     |    ab        | a_not_ab
    #      not a hits |    b_not_ab  | tot - ab  -a_not_b -b_not_a
    a_not_ab = a - ab
    b_not_ab = b - ab
    remainder = tot - ab - a_not_ab - b_not_ab
    contingency_table = [[ab, a_not_ab], [b_not_ab, remainder]]
    return contingency_table

def calc_ci_of_oddsratio(stats, p_vals, terms, d, run_type, addl_info):
    # ONLY run this for KM case. WHY ONLY FOR KM CASE?  RMS. Now can run for SKiM case too.
    key = "ci_of_oddsratio:" + str([terms[0]][0]) + "," + str([terms[1]][0])
    # stat based on ratios of the fet statistic (prior odds ratio)
    # First calculate the FET statistic based on the available data
    #  Then calculate the odds ratio for the two FET statistics.
    CI_LEVEL = 0.99  # RMS.  maybe make this a parameter.
    P_VAL_DISPLAY = round(1 - CI_LEVEL, 3)
    if CI_LEVEL == 0.99:
        Z_MULTIPLIER = 2.576
    elif CI_LEVEL == 0.95:
        Z_MULTIPLIER = 1.96
    elif CI_LEVEL == 0.999:
        Z_MULTIPLIER = 3.291
    else:
        print("need to specify a defined CI level")
        print("If this is a problem, please contact the developer. May want to make it a parameter")
        sys.exit(1)
    if run_type == "SKiM":
        # For Skim:
        '''
        Comments after talking with Yury on 6/28/2024
        We need to take the A-B link and the B-C link into account.
        1. We will calculate the fET p-value and odds ratio for the A-B link and the B-C link. For each A-B-C.
        2. We will then determine the BEST A-B-C link by multiplying the odds ratios for the A-B and B-C links.
        # note that multiplying the odds ratios together implies that the odds ratios are independent and that there
        # are no other paths in the graph that could confound the A-B-C link. This is a big assumption.
        3. We thought about  using the A-B or B-C FET p-value and odds ratio with the lower odds ratio (representing the weaker link)
           but this is problematic because the AB and BC sets may have different sizes/characteristics and the odds ratios may be skewed.
           For instance, the AB set may have a very high odds ratio because it is a small set, but the BC set may have a lower odds ratio because it is a larger set.
        For now (07/03/2024), we will use the sum of the log odds ratios for the A-B and B-C links to determine the best A-B-C link.
        4. We will use the best A-B-C1 and A-B-C2 links to calculate the confidence intervals for the best A-C1 and A-C2 links.
        Then determine if the CIs overlap or not.
        '''
        # a-b link,  go through all the a-b-c links and calculate the FET p-value and odds ratio for the A-B link
        #  use the skim lists
        best_dual_odds_ratio = []
        best_b_term = []
        best_c_term = []
        best_contingency_table1 = []
        best_contingency_table2 = []
        # RMS. Maybe make min_count_threshold a parameter.
        min_count_threshold = 2  # RMS.  This is a guess.  Maybe should be higher.   This requires that the AB counts and BC counts be at least this amount.
        for i in [0, 1]:
            best_dual_odds_ratio.append(0)
            best_b_term.append("")
            best_c_term.append("")
            best_contingency_table1.append([])
            best_contingency_table2.append([])
            for j, b in enumerate(d["skim_b_counts"][terms[i]]):
                ab_counts = d["skim_ab_counts"][terms[i]][j]
                contingency_table1 = build_standard_contingency_table(
                    a=d["skim_a_counts"][terms[i]][j],  # Total number of a hits
                    b=d["skim_b_counts"][terms[i]][j],  # Total number of b hits
                    ab=ab_counts,  # Number of ab hits
                    tot=d["tot_counts"][terms[i]][j],  # Total number of counts in db
                )
                odds_ratio_1, p_value_1, ab1, a_not_ab1, b_not_ab1, remainder1 = fet_for_whole_table(contingency_table1)
                bc_counts = d["intersection_counts"][terms[i]][j]
                contingency_table2 = build_standard_contingency_table(
                    a=d["skim_b_counts"][terms[i]][j],  # Total number of b hits
                    b=d["skim_c_counts"][terms[i]][j],  # Total number of c hits
                    ab=bc_counts,  # Number of bc hits
                    tot=d["tot_counts"][terms[i]][j],  # Total number of counts in db
                )
                odds_ratio_2, p_value_2, bc2, b_not_bc2, c_not_bc2, remainder2 = fet_for_whole_table(contingency_table2)
                total_sum_log_ORs = np.log(odds_ratio_1) + np.log(odds_ratio_2)
                if ab_counts > min_count_threshold and bc_counts > min_count_threshold:
                    if total_sum_log_ORs > best_dual_odds_ratio[i]:
                        best_dual_odds_ratio[i] = total_sum_log_ORs
                        best_b_term[i] = d["skim_b_terms"][terms[i]][j]
                        best_c_term[i] = d["skim_c_terms"][terms[i]][j]
                        best_contingency_table1[i] = [ab1, a_not_ab1, b_not_ab1, remainder1]
                        best_contingency_table2[i] = [bc2, b_not_bc2, c_not_bc2, remainder2]
        # Now we will calculate the dual odds ratio for the best A-B-C1 link vs the best A-B-C2 link
        CI_info = (
                         str(best_b_term[0]) + "-" + str(best_c_term[0]) + "(" + str(best_dual_odds_ratio[0]) + "):"
                         + str(best_b_term[1]) + "-" + str(best_c_term[1]) + "(" + str(best_dual_odds_ratio[1]) + ")"
                     )
        total_variance0 = np.divide(1, best_contingency_table1[0]).sum(0) + np.divide(1, best_contingency_table1[1]).sum(0)
        total_variance1 = np.divide(1, best_contingency_table2[0]).sum(0) + np.divide(1, best_contingency_table2[1]).sum(0)
        upperCI0 = best_dual_odds_ratio[0] + Z_MULTIPLIER * np.sqrt(total_variance0)
        lowerCI0 = best_dual_odds_ratio[0] - Z_MULTIPLIER * np.sqrt(total_variance0)
        upperCI1 = best_dual_odds_ratio[1] + Z_MULTIPLIER * np.sqrt(total_variance1)
        lowerCI1 = best_dual_odds_ratio[1] - Z_MULTIPLIER * np.sqrt(total_variance1)
        sep_count = 0
        if lowerCI0 > upperCI1 or upperCI0 < lowerCI1:
            if upperCI0 < lowerCI1:
                separation = lowerCI1 - upperCI0
            else:  # lowerCI0 > upperCI1:
                separation = lowerCI0 - upperCI1
        else:  # lowerCI0 <= upperCI1 or upperCI0 >= lowerCI1:  #  CIs overlap
            separation1 = "X"
            separation2 = "X"
            if lowerCI0 <= upperCI1:
                separation1 = lowerCI0 - upperCI1
                sep_count += 1
            if upperCI0 >= lowerCI1:
                separation2 = lowerCI1 - upperCI0
                sep_count += 1
            if sep_count > 1:
                separation = max(separation1, separation2)  # This is the separation of the CIs.
                # we want max here, because we want the minimal separation and both numbers are negative.
            else:
                if separation1 == "X":
                    separation = separation2
                else:
                    separation = separation1
        stats[key] = separation
    else:  # KM
        # contingency table to get odds ratios
        contingency_table1 = build_standard_contingency_table(
            a=d["km_a_counts"][terms[0]][0],  # Total number of a hits for 1st term
            b=d["km_b_counts"][terms[0]][0],  # Total number of b hits for 1st term
            ab=d["intersection_counts"][terms[0]][0],  # Number of ab hits for 1st term
            tot=d["tot_counts"][terms[0]][0],  # Total number of counts in db for 1st term
        )
        orig_odds_ratio_1, orig_p_value_1, ab1, a_not_ab1, b_not_ab1, remainder1 = fet_for_whole_table(contingency_table1)
        contingency_table2 = build_standard_contingency_table(
            a=d["km_a_counts"][terms[1]][0],  # Total number of a hits for 2nd term
            b=d["km_b_counts"][terms[1]][0],  # Total number of b hits for 2nd term
            ab=d["intersection_counts"][terms[1]][0],  # Number of ab hits for 2nd term
            tot=d["tot_counts"][terms[1]][0],  # Total number of counts in db for 2nd term
        )
        orig_odds_ratio_2, orig_p_value_2, ab2, a_not_ab2, b_not_ab2, remainder2 = fet_for_whole_table(contingency_table2)
        stats[key] = orig_odds_ratio_1 / orig_odds_ratio_2
        # getting CIs of odds ratios
        upperCI0 = orig_odds_ratio_1 + Z_MULTIPLIER * np.sqrt(1 / ab1 + 1 / a_not_ab1 + 1 / b_not_ab1 + 1 / remainder1)
        lowerCI0 = orig_odds_ratio_1 - Z_MULTIPLIER * np.sqrt(1 / ab1 + 1 / a_not_ab1 + 1 / b_not_ab1 + 1 / remainder1)
        upperCI1 = orig_odds_ratio_2 + Z_MULTIPLIER * np.sqrt(1 / ab2 + 1 / a_not_ab2 + 1 / b_not_ab2 + 1 / remainder2)
        lowerCI1 = orig_odds_ratio_2 - Z_MULTIPLIER * np.sqrt(1 / ab2 + 1 / a_not_ab2 + 1 / b_not_ab2 + 1 / remainder2)
        CI_info = " NEED to add CIINFO for KM here."
    overlapStr = " CIs (" + CI_info + ") OVERLAP. NOT SIGNIFICANT at CI LEVEL:" + str(CI_LEVEL)
    pval = 1
    if lowerCI0 > upperCI1 or upperCI0 < lowerCI1:
        overlapStr = " NO OVERLAP of CIs (" + CI_info + ").  SIGNIFICANT at CI LEVEL:" + str(CI_LEVEL)
        pval = P_VAL_DISPLAY
    p_vals[key] = pval
    addl_info[key] = (
        "Doing ratio based on FET STAT (not pvalue), reporting pval of 1 for overlap of CIs, pval of " + str(P_VAL_DISPLAY) + " for no overlap of CIs."
        + " CI1: "
        + str(lowerCI0)
        + "-"
        + str(upperCI0)
        + "    CI2: "
        + str(lowerCI1)
        + "-"
        + str(upperCI1)
        + overlapStr
    )
    return stats, p_vals, addl_info

def fet_for_whole_table(contingency_table):
    # Construct the contingency table
    #                     b hits        not b_hits
    #                   --------     --------
    #      a hits     |    ab        | a_not_ab
    #      not a hits |    b_not_ab  | tot - ab  -a_not_b -b_not_a
    #  RMS.  I think this contingency table is correct.
    # a_not_ab = a - ab
    # b_not_ab = b - ab
    # remainder = tot - ab - a_not_ab - b_not_ab
    # contingency_table = [[ab, a - ab], [b - ab, remainder]]
    # Perform Fisher's exact test
    odds_ratio, p_value = fisher_exact(contingency_table)
    ab = contingency_table[0][0]
    a_not_ab = contingency_table[0][1]
    b_not_ab = contingency_table[1][0]
    remainder = contingency_table[1][1]
    return odds_ratio, p_value, ab, a_not_ab, b_not_ab, remainder

def check_for_no_or_only1_line(sorted_lines, unique_terms, sort_index):
    terms = ""
    stats = {}  # statistics keyed by type
    p_vals = {}  # p_vals keyed by type
    addl_info = {}  # additional statistical info keyed by type
    if len(sorted_lines) == 0:
        terms = terms
        stats["NOLINES"] = "NA"
        # terms = "NONE"
        # stats["NOLINES"] = terms
        p_vals["NOLINES"] = "NA"
        addl_info["NOLINES"] = ""
        return terms, stats, p_vals, addl_info

    if len(sorted_lines) == 1 or len(unique_terms) == 1:
        terms = sorted_lines[0][sort_index]
        stats["ONLY1"] = terms
        p_vals["ONLY1"] = "NA"
        addl_info["ONLY1"] = ""
        return terms, stats, p_vals, addl_info
    return terms, stats, p_vals, addl_info

def filter_and_sort_lines(file_ctr, lines, fet_ab, fet_bc, lines_prior_year, in_terms, out_f_kept):
    # set sort positions and remove lines that don't meet fet cutoffs
    ab_pval_index = 8
    kept_lines = []
    if file_ctr == 1:  # KM Case
        # print("KM:", lines[0][0])
        sort_index = 3
        for line in lines:
            if float(line[ab_pval_index]) < fet_ab and line[sort_index] in in_terms:
                kept_lines.append(line)
    else:
        # print("SKiM", lines[0][0])
        sort_index = 16
        bc_pval_index = 14
        for line in lines:
            if float(line[ab_pval_index]) < fet_ab and float(line[bc_pval_index]) < fet_bc and line[sort_index] in in_terms:
                kept_lines.append(line)
    sorted_lines = sorted(kept_lines, key=lambda x: x[sort_index])
    sorted_prior_year_lines = sorted(lines_prior_year, key=lambda x: x[sort_index])
    unique_terms = {}
    for line in sorted_lines:
        out_f_kept.write("\t".join(line) + "\n")
        unique_terms[line[sort_index]] = "X"
    return sorted_lines, sorted_prior_year_lines, kept_lines, unique_terms, sort_index

def reorder_dict(d, key_order):
    new_dict = OD()
    for key in key_order:
        if key in d:
            new_dict[key] = d[key]
    for key in d.keys():
        if key not in new_dict:
            new_dict[key] = d[key]
    return new_dict

def reorder_nested_dict(d, key_order):
    new_dict = OD()
    for top_key in d.keys():
        new_dict[top_key] = OD()
        for key in key_order:
            if key in d[top_key]:
                new_dict[top_key][key] = d[top_key][key]
        for key in d[top_key].keys():
            if key not in new_dict[top_key]:
                new_dict[top_key][key] = d[top_key][key]
    return new_dict

def reorder_terms(terms, key_order):
    new_terms = []
    for key in key_order:
        if key in terms:
            new_terms.append(key)
    for key in terms:
        if key not in new_terms:
            new_terms.append(key)
    return new_terms

if __name__ == "__main__":
    main()
