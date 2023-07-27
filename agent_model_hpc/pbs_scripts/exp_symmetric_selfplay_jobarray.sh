#!/bin/bash
#PBS -N exp_symmetric_selfplay_jobarray
#PBS -l ncpus=1
#PBS -l mem=400mb
#PBS -l walltime=04:00:00
#PBS -m ea
#PBS -WMail_Users=__USER_EMAIL__
#PBS -o /scratch/__USER__/obs_exp/pbs_output/
#PBS -e /scratch/__USER__/obs_exp/pbs_output/
#PBS -J 0-5:1

trap "echo "---------------------"$'\r'; qstat -f $PBS_JOBID" EXIT

IFS='[' read -ra JOBID_ARRAY <<< $PBS_JOBID
PARENT_JOBID=${JOBID_ARRAY[0]}
REMAINDER_JOBID_ARRAY=${JOBID_ARRAY[1]}

IFS=']' read -ra SUB_JOBID_ARRAY <<< $REMAINDER_JOBID_ARRAY
SUB_JOBID=${SUB_JOBID_ARRAY[0]}

JOBID=${PARENT_JOBID}_${SUB_JOBID}
JOB_DIR=$JOBID

RESULTS_DIR=~/__BASEPATH__/results/experiments/symmetric_selfplay/$PARENT_JOBID/$JOB_DIR
echo "SHELL:: Creating RESULTS_DIR: " $RESULTS_DIR
mkdir -p $RESULTS_DIR


SCRATCH_DIR=/scratch/$USER
SCRATCH_JOB_DIR=$SCRATCH_DIR/$JOBID
echo "SHELL:: Creating SCRATCH_JOB_DIR: " $SCRATCH_JOB_DIR
mkdir $SCRATCH_JOB_DIR


# create journal job dir in USER HOME
JOURNAL_DEST=~/__BASEPATH__/results/obs_exp/journal/$PARENT_JOBID/sj_summary
mkdir -p $JOURNAL_DEST

# create (if not exist) obs_exp/heartbeat and obs_exp/journal on SCRATCH
SCRATCH_OBS_EXP_HEARTBEAT=$SCRATCH_DIR/obs_exp/heartbeat
mkdir -p $SCRATCH_OBS_EXP_HEARTBEAT

SCRATCH_OBS_EXP_JOURNAL=$SCRATCH_DIR/obs_exp/journal
mkdir -p $SCRATCH_OBS_EXP_JOURNAL


module load EasyBuild/4.2.2
module load Python/3.7.4-GCCcore-8.3.0

cd ~/virtualenvs/agent_model/bin
source activate

cd ${PBS_O_WORKDIR}

# 6, 26, 17, 14
strategy_set_gametheoretic=("allc" "alld" "bully_naive" "fictitiousplay" "random" "tft")
strategy_set_kbfas=("kbfa_00" "kbfa_01" "kbfa_02" "kbfa_03" "kbfa_04" "kbfa_05" "kbfa_06" "kbfa_07" "kbfa_08" "kbfa_09" "kbfa_10" "kbfa_11" "kbfa_12" "kbfa_13" "kbfa_14" "kbfa_15" "kbfa_16" "kbfa_17" "kbfa_18" "kbfa_19" "kbfa_20" "kbfa_21" "kbfa_22" "kbfa_23" "kbfa_24" "kbfa_25")
strategy_set_bandits=("bandit_inc_softmax_ap_2ed" "bandit_noninc_softmax_ap_2ed" "bandit_pursuit_sav" "bandit_reinfcomp" "bandit_sav_inc" "bandit_sav_inc_optimistic_greedy" "bandit_sav_inc_softmax" "bandit_sav_noninc" "bandit_sav_noninc_softmax" "bandit_sl_direct" "bandit_sl_la_lri" "bandit_sl_la_lrp" "bandit_wa" "bandit_wa_optimistic_greedy" "bandit_wa_softmax" "bandit_wa_softmax_ap_2ed" "bandit_wa_ucb")
strategy_set_rlmethods=("actor_critic_1ed" "actor_critic_1ed_eligibility_traces" "actor_critic_1ed_replacetrace" "double_qlearning" "expected_sarsa" "qlearning" "rlearning" "sarsa" "sarsa_lambda" "sarsa_lambda_replacetrace" "watkins_naive_q_lambda" "watkins_naive_q_lambda_replacetrace" "watkins_q_lambda" "watkins_q_lfa" )


python3 -m run_exp.exp_symmetric_selfplay -j $PBS_JOBID -w $SCRATCH_DIR -g pd -r scalar -s ${strategy_set_gametheoretic[$PBS_ARRAY_INDEX]} -e 500 -t 1000 -p na -k false -z true -c true -l false

cd $SCRATCH_DIR || exit 1

ARCHIVE_FILENAME=$JOBID.tar.gz
echo "SHELL:: creating archive: " $ARCHIVE_FILENAME
tar --remove-files --create --gzip --file=$ARCHIVE_FILENAME -C $SCRATCH_DIR $JOB_DIR

echo "SHELL:: moving archive ..."
mv -fv $ARCHIVE_FILENAME $RESULTS_DIR

# remove heartbeat and journal files
HEARTBEAT_FILE=exp_$JOBID.heartbeat
JOURNAL_FILE=exp_${JOBID}_summary.json

echo "SHELL:: deleting heartbeat file from SCRATCH: " $HEARTBEAT_FILE
rm -f $SCRATCH_OBS_EXP_HEARTBEAT/$HEARTBEAT_FILE

echo "SHELL:: moving journal file from SCRATCH to USER HOME: " $JOURNAL_FILE $JOURNAL_DEST
mv -fv $SCRATCH_OBS_EXP_JOURNAL/$JOURNAL_FILE $JOURNAL_DEST


echo "SHELL:: Finished."
