package org.creamycookie

import kotlin.io.path.Path

val PATH = Path("training_data.csv.gz")
val MODE = Mode.ALL

const val START_FRESH = false
const val START_NUMBER_HILL_CLIMBING = false
const val ONLY_MUTATE_NUMBERS = false

// train with a representative sample to speed up the process
const val NUM_TRAINING_DATA = 32_768 // total: 10_299_575
const val NUM_VALIDATION_PER_TYPE = 32_768 * 3
const val USE_ALL_FOR_VALIDATION_DATA = true

const val MAX_VALUE = 32_767.0
const val FIX_MAX_OVERLAP_MS = 358.0

const val MAX_MS_CONSIDERED = 100_000.0

const val FMT_DECIMAL_PLACES = 3
const val PRINT_FOR_JAVA = true

const val ONLY_CHOSEN_FROM_INITIAL_POPULATION = false

// When the keyboard starts, there is no previous keypress, so what is a good
// initial value for something like prev_up_th_down_dur? Use this to find out.
// It only makes sense to do so, when you have already found a good solution.
const val START_FINDING_BEST_INITIAL_TRAIN_COL_VALUE = false
const val USE_BINARY_SEARCH_TO_FIND_BEST_INITIAL_TRAIN_COL_VALUE = false
val FINDING_BEST_INITIAL_TRAIN_COL = TrainCol.PREV_UP_TH_DOWN_DUR
const val FINDING_BEST_INITIAL_TRAIN_COL_FROM = 0
const val FINDING_BEST_INITIAL_TRAIN_COL_TO = 200

// use random sections of the training data for the fitness calculation
const val UNSTABLE_SAMPLING = false
const val UNSTABLE_SAMPLING_SIZE = 8_192