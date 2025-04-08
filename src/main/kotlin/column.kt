package org.creamycookie

enum class InputTrainCol {
    PREV_DOWN_TIME,
    PREV_UP_TIME,
    TH_DOWN_TIME,
    NEXT_DOWN_TIME,
    NEXT_UP_TIME,
    TH_UP_TIME,
    LAST_DOWN_TIME,
    PREV_IS_MOD,
    IS_MOD,
}


enum class TrainCol(
    // only cases like tap hold down, next down, next up
    val onlyWrapped: Boolean = false,

    // only cases like tap hold down, next down, last down
    val onlyTripleDown: Boolean = false,

    val secret: Boolean = false,
    val isDur: Boolean = false
) {
    PREV_IS_MOD,
    PREV_DUR(isDur = true),
    PREV_UP_TH_DOWN_DUR(isDur = true),
    TH_DOWN_NEXT_DOWN_DUR(isDur = true),
    NEXT_DOWN_TIME(secret = true),
    NEXT_UP_TIME(secret = true),
    TH_UP_TIME(secret = true),
    IS_MOD(secret = true),
    NEXT_DUR(onlyWrapped = true, isDur = true),
    TH_DOWN_NEXT_UP_DUR(onlyWrapped = true, isDur = true),
    LAST_DOWN_TIME(secret = true),
    NEXT_DOWN_LAST_DOWN_DUR(onlyTripleDown = true, isDur = true),
}


enum class Mode {
    ALL, WRAPPED, TRIPLE_DOWN
}