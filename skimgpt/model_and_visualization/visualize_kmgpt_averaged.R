# visualize kmgpt (averaged over iterations)

# open libraries
library(ggplot2)
library(ggpubr)
library(grid)
library(viridisLite)
library(gridExtra)
library(dplyr)
library(stringr)
library("optparse")
library(patchwork)
library(lemon)

# make arguments
option_list = list(
  make_option(c("-i", "--datatype"), type="character", default=NULL, 
              help="km or skim (what file to look for? \ 
              either km_hyp_stats.txt or skim_hyp_stats.txt)", 
              metavar="character"),
  make_option(c("-p", "--projpath"), type="character", default=NULL, 
              help="path where input file is (km_with_gpt_wrapper_results.tsv)", 
              metavar="character"),
  make_option(c("-d", "--discover"), type="character", default=NULL, 
              help="discover date of hypothesis", 
              metavar="character"),
  make_option(c("-a", "--accept"), type="character", default=NULL, 
              help="acceptance date of hypothesis", 
              metavar="character"),
  make_option(c("-x", "--x_date"), type="character", default=NA, 
              help="acceptance date of hypothesis", 
              metavar="character"),
  make_option(c("-l", "--labels"), type="character", default=NULL,
              help="Comma-separated list of term labels for legend \
              (e.g., 'microbiome,vaccines')", metavar="character"),
  make_option(c("-t", "--title"), type="character", default=NULL, 
              help="title of plot", 
              metavar="character"),
  make_option(c("-k", "--labels2"), type="character", default=NULL,
              help="Comma-separated list of labels for discovery and acceptance \
              (e.g., 'discover,accept')", metavar="character"),
  make_option(c("-m", "--move"), type="character", default=NULL,
              help="move discovery/acceptance labels. Comma separated list of4 \
              numbers required: x, y for discovery, x, y for acceptance \
              e.g. -m '0.1,0.1,0.1,0.1'", metavar="character"), # if neg.: --move=-0.5,-0.5,-0.5,-0.5
  make_option(c("-n", "--xinterval"), type="integer", default=1,
              help="Interval for x-axis labels (e.g., 2 for every 2nd year)",
              metavar="integer")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
datatype <- opt$datatype
projPath <- opt$projpath
d_date <- opt$discover
a_date <- opt$accept
x_date <- opt$x_date
titleX <- opt$title
if (!is.null(opt$labels)) {
  labelx <- unlist(strsplit(opt$labels, ","))
  labelx <- trimws(labelx)
}
if (!is.null(opt$labels2)) {
  labelx2 <- unlist(strsplit(opt$labels2, ","))
  labelx2 <- trimws(labelx2)
} else {
  labelx2 <- c("discover", "acceptance")
}
if (!is.null(opt$move)) {
  movex <- unlist(strsplit(opt$move, ","))
  movex <- trimws(movex)
  movex <- as.numeric(movex)
} else {
  movex <- c(-0.1, 1, -0.05, 1)
}
x_interval <- opt$xinterval

# for testing
# datatype <- "km"
# projPath <- "/Users/bmoore/Desktop/StewartLab_projects/Hyp1vsHyp2_paper/KM-GPT_noncomparitive/output_20260916112337_cy_range1975-2026_cy_inc_1_cy_depth_5_iterations10_scrapie/"
# d_date <- "1982"
# a_date <- "1997"
# #x_date <- "2006"
# titleX <- 'KM-GPT Scrapie: Protein vs. Virus 1975-2026'
# # 'KM-GPT Cervical cancer: HPV vs HSV 1975-2025'
# # 'KM-GPT Peptic ulcer: BI vs Stress 1975-2025'
#   # 'KM-GPT Autism: Genetics vs. Vaccines 1975-2026'
#   # 'KM-GPT Scrapie: Protein vs. Virus 1975-2026'
#   # 'KM-GPT Cardiovascular disease: Statins vs. Hormones 1975-2026'
# labelx2 <- c("proposed", "rejected") #"discover", "accepted", "reconsidered", "rejected"
# movex <- c(-0.1, 1, -0.05, 1)
# x_interval <- 5

# get data_dir and output
setwd(projPath)
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
output <- paste0("./output_visualization_", timestamp)
print(output)
dir.create(output, mode = "0777", showWarnings = FALSE)
output <- paste0(output, "/")

filename <- "km_with_gpt_wrapper_results.tsv"
# read in data
if(datatype=="km"){
  km_data = read.csv2(filename, header=TRUE, sep="\t")
}

# convert score column to numeric (handles "N/A" strings -> NA)
km_data <- km_data %>%
  mutate(gpt_5.6_terra_score = na_if(gpt_5.6_terra_score, "N/A"),
         gpt_5.6_terra_score = as.numeric(gpt_5.6_terra_score))

# ---- NEW: average score across iterations for each year/hypothesis ----
# Your data now has one row per (censor_year, Hypothesis, iter_number).
# Collapse iterations down to one row per (censor_year, Hypothesis) by
# taking the mean score, and keep sd/n around in case you want error bars.
km_data_avg <- km_data %>%
  group_by(censor_year, Hypothesis) %>%
  summarise(
    mean_score = mean(gpt_5.6_terra_score, na.rm = TRUE),
    sd_score   = sd(gpt_5.6_terra_score, na.rm = TRUE),
    n_iter     = sum(!is.na(gpt_5.6_terra_score)),
    .groups = "drop"
  ) %>%
  mutate(
    se_score = sd_score / sqrt(n_iter)
  )

# use the averaged data from here on
km_data <- km_data_avg

# plot
color_vector <- c("gold","#433E85FF")
xc <- "2022"
yc2 <- 1.5
all_dates <- sort(unique(km_data$censor_year))
date_breaks <- all_dates[seq(1, length(all_dates), by = x_interval)]

p1 <- ggplot(km_data, aes(factor(censor_year), mean_score)) + 
  geom_line(aes(group = factor(Hypothesis), colour = factor(Hypothesis)), lineend="round") +
  scale_colour_manual(name= "Term", values = color_vector) +
  scale_x_discrete(breaks = date_breaks, labels = date_breaks) +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        legend.position="right",
        panel.grid.minor = element_blank()) +
  ylab("Mean score") + xlab("Year") +
  geom_vline(xintercept=d_date, linetype = 2, colour = "brown") +
  annotate("text",label=labelx2[1],y=yc2, x=d_date, hjust=movex[1], 
           vjust=movex[2], colour = "brown", angle = 90) +
  geom_vline(xintercept=a_date, linetype = 2, colour = "black") +
  annotate("text",label=labelx2[2],y=yc2, x=a_date, hjust=movex[3], 
           vjust=movex[4], colour = "black", angle = 90) 

p1

p2 <- ggplot(km_data, aes(factor(censor_year), mean_score)) + 
  geom_point(aes(group = factor(Hypothesis), colour = factor(Hypothesis))) +
  # optional: show spread across iterations as error bars (comment out if not wanted)
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score,
                     colour = factor(Hypothesis)), width = 0.2, alpha = 0.6) +
  scale_colour_manual(name= "Term", values = color_vector) +
  scale_x_discrete(breaks = date_breaks, labels = date_breaks) +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        legend.position="right",
        panel.grid.minor = element_blank()) +
  ylab("Mean score") + xlab("Year") +
  geom_vline(xintercept=d_date, linetype = 2, colour = "brown") +
  annotate("text",label=labelx2[1],y=yc2, x=d_date, hjust=movex[1], 
           vjust=movex[2], colour = "brown", angle = 90) +
  geom_vline(xintercept=a_date, linetype = 2, colour = "black") +
  annotate("text",label=labelx2[2],y=yc2, x=a_date, hjust=movex[3], 
           vjust=movex[4], colour = "black", angle = 90) 

p2

p3 <- ggplot(km_data, aes(factor(censor_year), mean_score)) + 
  geom_pointline(aes(group = factor(Hypothesis), colour = factor(Hypothesis))) +
  geom_errorbar(aes(ymin = mean_score - se_score, ymax = mean_score + se_score,
                    colour = factor(Hypothesis)), width = 0.2, alpha = 0.6) +
  scale_colour_manual(name= "Term", values = color_vector) +
  scale_x_discrete(breaks = date_breaks, labels = date_breaks) +
  theme_bw() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1),
        legend.position="right",
        panel.grid.minor = element_blank()) +
  ylab("Mean score") + xlab("Year") +
  geom_vline(xintercept=d_date, linetype = 2, colour = "brown") +
  annotate("text",label=labelx2[1],y=yc2, x=d_date, hjust=movex[1], 
           vjust=movex[2], colour = "brown", angle = 90) +
  geom_vline(xintercept=a_date, linetype = 2, colour = "black") +
  annotate("text",label=labelx2[2],y=yc2, x=a_date, hjust=movex[3], 
           vjust=movex[4], colour = "black", angle = 90) 

p3

# add third vline if needed
if (!is.na(x_date)){
  p1 <- p1 + geom_vline(xintercept=x_date, linetype = 2, colour = "grey") +
    annotate("text",label=labelx2[3],y=yc2, x=x_date, hjust=movex[3], 
             vjust=movex[4], colour = "black", angle = 90) 
  p2 <- p2 + geom_vline(xintercept=x_date, linetype = 2, colour = "grey") +
    annotate("text",label=labelx2[3],y=yc2, x=x_date, hjust=movex[3], 
             vjust=movex[4], colour = "black", angle = 90) 
  p3 <- p3 + geom_vline(xintercept=x_date, linetype = 2, colour = "grey") +
    annotate("text",label=labelx2[3],y=yc2, x=x_date, hjust=movex[3], 
             vjust=movex[4], colour = "black", angle = 90)
}

full_path <- paste0(projPath,output)
wrapped_title <- str_wrap(titleX, width = 80)  # Adjust width as needed
# make pdf
pdf(file=paste0(output, "KM_GPT_scores_line.pdf"),
    width=7, height=5)
  p1 +plot_annotation(
  title = wrapped_title,
  caption = full_path)
dev.off()

pdf(file=paste0(output, "KM_GPT_scores_points.pdf"),
    width=7, height=5)
  p2 +plot_annotation(
  title = wrapped_title,
  caption = full_path)
dev.off()

pdf(file=paste0(output, "KM_GPT_scores_point-lines.pdf"),
    width=7, height=5)
  p3 +plot_annotation(
  title = wrapped_title,
  caption = full_path)
dev.off()

# parameter list

B1_term <- unique(km_data$Hypothesis)[1]
B2_term <- unique(km_data$Hypothesis)[2]
parameter_list = c(filename, full_path, titleX, d_date,
                   a_date, B1_term, B2_term)
print(parameter_list)
# Create dataframe from parameter list
parameter_df <- data.frame(
  Parameter = c("filename", "ProjectPath", "title", "discover_date", "acceptance_date", 
                "term_B1", "term_B2"),
  Value = parameter_list
)

# Write parameter dataframe to CSV
write.csv(parameter_df, file = paste0(output, "parameters.csv"), row.names = FALSE)

# also write out the averaged data itself, for your own records/checking
write.csv(km_data, file = paste0(output, "averaged_scores_by_year_hypothesis.csv"), row.names = FALSE)

# write packages
writeLines(capture.output(sessionInfo()), paste0(output,"sessionInfo.txt"))

