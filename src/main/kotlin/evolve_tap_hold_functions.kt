package org.creamycookie

import io.jenetics.*
import io.jenetics.engine.Engine
import io.jenetics.engine.EvolutionResult
import io.jenetics.ext.SingleNodeCrossover
import io.jenetics.ext.TreeRewriteAlterer
import io.jenetics.ext.rewriting.TreeRewriteRule
import io.jenetics.ext.util.Tree
import io.jenetics.ext.util.TreeNode
import io.jenetics.prog.MathRewriteAlterer
import io.jenetics.prog.ProgramChromosome
import io.jenetics.prog.ProgramGene
import io.jenetics.prog.op.*
import io.jenetics.prog.regression.Complexity
import io.jenetics.prog.regression.Error
import io.jenetics.prog.regression.Regression
import io.jenetics.prog.regression.Sampling
import io.jenetics.util.ISeq
import io.jenetics.util.RandomRegistry
import org.creamycookie.utils.*
import java.time.Instant
import java.util.*
import java.util.stream.Collectors
import java.util.stream.Stream
import java.util.zip.GZIPInputStream
import kotlin.io.path.inputStream
import kotlin.math.abs
import kotlin.math.absoluteValue
import kotlin.math.max
import kotlin.math.min
import kotlin.text.Charsets.UTF_8


private const val COMMENT_LINE_PREFIX = "        // "
private const val FUNCTION_LINE_PREFIX = "        "


// MathOp.DIV -> can lead to divide by zero
private val safeDiv: Op<Double> = Op.of("sd") { a, b ->
    if (b == 0.0) a else a / b
}

val divToSd = TreeRewriteRule.parse("div(${'$'}x,${'$'}y) -> sd(${'$'}x,${'$'}y)") {
    if (it == "sd") safeDiv else MathOp.toMathOp(it)
}!!

private val ops: ISeq<Op<Double>> =
    ISeq.of(MathOp.ADD, MathOp.SUB, MathOp.MUL, MathOp.MAX, MathOp.MIN, MathOp.ABS, safeDiv)

private val vars = ISeq.of(
    TrainCol.entries
        .filter {
            !it.secret
                    && (MODE == Mode.WRAPPED || !it.onlyWrapped)
                    && (MODE == Mode.TRIPLE_DOWN || !it.onlyTripleDown)
        }
        .map { Var.of<Double>(it.name.lowercase(), it.ordinal) }
)

private val terminals: ISeq<Op<Double>> = ISeq.concat(
    vars,
    Const.of(0.0),
    Const.of(1.0),
    Const.of(2.0),
    Const.of(3.0),
    Const.of(5.0),
    Const.of(8.0),
    Const.of(0.6180339887498948), // inverse of Phi
    Const.of(0.00001),
    EphemeralConst.of { RandomRegistry.random().nextGaussian(0.0, 1000.0) },
    EphemeralConst.of { RandomRegistry.random().nextGaussian(0.0, 500.0) },
    EphemeralConst.of { RandomRegistry.random().nextGaussian(0.0, 250.0) },
    EphemeralConst.of { RandomRegistry.random().nextGaussian(0.0, 100.0) },
    EphemeralConst.of { RandomRegistry.random().nextGaussian(0.0, 50.0) },
)

private val sampling = if (MODE == Mode.ALL) sampling(::estimateOverlap) else sampling(::guessIfIsHold)


private inline fun sampling(crossinline fit: (Array<Double>, Double) -> Double): Sampling<Double> {
    if (UNSTABLE_SAMPLING) {
        return Sampling { programm ->
            val rng = RandomRegistry.random()
            val s = rng.nextInt(trainingData.size - UNSTABLE_SAMPLING_SIZE)

            val calculated = (s..<(s + UNSTABLE_SAMPLING_SIZE)).sumOf {
                val ev = trainingData[it]
                fit(ev, programm.callWith(ev))
            }

            Sampling.Result(arrayOf(calculated), results)
        }
    }

    return Sampling { programm ->
        val calculated = trainingData.sumOf {
            fit(it, programm.callWith(it))
        }

        Sampling.Result(arrayOf(calculated), results)
    }
}

fun Tree<out Op<Double>, *>.callWith(ev: Array<Double>): Double {
    return reduce(ev) { acc, p -> acc.apply(p) }!!
}

// This punishes programs that become too large.
// A large value leads to only a tiny penalty for the size.
// We make the value so large, since we generally want the better program to win.
// Only if all other things are equal, we want the smaller one to beat the larger one.
private val complexity = Complexity.ofNodeCount<Double>(512)!!

private val regression = Regression.of(
    Regression.codecOf(ops, terminals, 6) { it.gene().size() <= 64 },
    Error.of(
        { calculated: Array<Double>, _: Array<Double> ->
            //LossFunction.mse(calculated, expected)
            //calculated.average()
            calculated[0]
        },
        complexity
    ),
    sampling
)

private var validationData = mutableListOf<Array<Double>>()
private val trainingData = readTrainingData()

private val results = arrayOf(0.0)

// # minimums
// - prev_dur = 1
// - prev_up_th_down_dur = -5120
// - th_down_next_down_dur = 0
// - next_dur = 1
// - th_down_next_up_dur = 1
// - next_down_last_down_dur = 0
//
// # maximums
// - prev_dur = 47595
// - prev_up_th_down_dur = 166204437
// - th_down_next_down_dur = 8800
// - next_dur = 6823
// - th_down_next_up_dur = 8952
// - next_down_last_down_dur = 18520655
private val initialPopulation = listOfNotNull(
    // mod: 909842 / 1319001, non-mod: 8947374 / 8980574
    // ~Correct:    95.705 % (of 10,299,573)
    // Node Count:  23
    parseToProgGenotypeIfMode(
        Mode.ALL,
        "abs(max(max(3.0614198142811473, ((max(-136.16210285390332*prev_up_th_down_dur - 315.5284051935217, 1386.754557078502) - max(prev_up_th_down_dur, 325.5094909990815)) - 6.423200405547902*th_down_next_down_dur) + 302.953209210422), neg(prev_up_th_down_dur)))",
        chosen = true
    ),

    // There have been slightly better performing solutions, but when trying
    // them on a different data set, they would perform worse.
    // Simpler solutions tend to generalize better.
    // mod: 320852 / 343898, non-mod: 109651 / 131951
    // ~Correct:    90.470 % (of 475,847)
    // Node Count:  34
    parseToProgGenotypeIfMode(
        Mode.WRAPPED,
        "sd(sd(max(prev_up_th_down_dur * 0.6423792237402978, 23.45217822310148), max(1.1125613676009913 + sd(558.607953494396, th_down_next_down_dur), (max(max(1.1125613676009913, -5.763034442713599*prev_up_th_down_dur - 184.2279499410631)*-prev_up_th_down_dur, 110.87524870231138) + sd(4170.020640954126, next_dur)) - 179.26977975231142)), sd(542.2182734892068, th_down_next_down_dur))",
        chosen = true
    ),

    // mod: 16938 / 28959, non-mod: 251280 / 251715
    // ~Correct:    95.562 % (of 280,672)
    // Node Count:  26
    parseToProgGenotypeIfMode(
        Mode.TRIPLE_DOWN,
        "sd(abs(-0.029755300916950864*prev_up_th_down_dur - 9.283691710426819) + (0.2559898995352028 + prev_is_mod*(0.01803255250147312*max(-prev_up_th_down_dur, 17.524751405626137)))*th_down_next_down_dur, max(-prev_up_th_down_dur, 14.022869925365338)*-9.263839511624653)",
        chosen = true
    ),

    // C float version of previous - performed the same (this was true for the other modes as well)
    parseToProgGenotypeIfMode(
        Mode.TRIPLE_DOWN,
        "sd(abs(-0.029755301773548126*prev_up_th_down_dur - 9.28369140625) + (0.2559899091720581 + (0.018032552674412727*max(-prev_up_th_down_dur, 17.524751663208008))*prev_is_mod)*th_down_next_down_dur, max(-prev_up_th_down_dur, 14.022870063781738)*-9.263839721679688)",
    ),
)

var chosenGenotype: Genotype<ProgramGene<Double>>? = null

fun parseToProgGenotypeIfMode(mode: Mode, expr: String, chosen: Boolean = false): Genotype<ProgramGene<Double>>? {
    if (mode != MODE) return null
    val geno = parseToProgGenotype(expr)
    if (chosen) {
        chosenGenotype = geno
    } else if (ONLY_CHOSEN_FROM_INITIAL_POPULATION) {
        return null;
    }
    return geno
}


fun parseToProgGenotype(expr: String): Genotype<ProgramGene<Double>> {
    // MathExpr doesn't know about sd function
    val patchedExpr = expr.replace("sd(", "div(")
    val tree: Tree<Op<Double>, *> = MathExpr.parse(patchedExpr).tree()

    val asNode: TreeNode<Op<Double>> = TreeNode.ofTree(tree)
    Var.reindex(asNode, vars.associateWith { it.index() })

    // now we can turn it back to sd
    divToSd.rewrite(asNode)

    // this was .of(tree
    return Genotype.of(ProgramChromosome.of(asNode, ops, terminals))
}


inline fun <T> Array<T>.get(getter: () -> TrainCol): T {
    return this[getter().ordinal]
}


fun howToRunProgram() {
    val treeNode = parseToProgGenotype("sd(3, -1.5)").gene().toTreeNode()
    val input = trainingData[0]
    val result = treeNode.callWith(input)
    println("$treeNode called with ${input.contentToString()} = $result")
}

fun <T> Stream<T>.toMutableList(): MutableList<T> {
    return collect(Collectors.toCollection { mutableListOf() })
}

private fun readTrainingData(): List<Array<Double>> {
    GZIPInputStream(PATH.inputStream()).bufferedReader(UTF_8).use { reader ->
        reader.readLine() // skip CSV heading

        var count = 0
        val minValues = Array(TrainCol.entries.size) { Double.POSITIVE_INFINITY }
        val maxValues = Array(TrainCol.entries.size) { Double.NEGATIVE_INFINITY }

        val result = reader.lines()
            .map {
                count += 1
                val parts = it.split("\t").map(String::toDouble)
                val res = Array(TrainCol.entries.size) { 0.0 }

                res[TrainCol.PREV_IS_MOD.ordinal] = parts[InputTrainCol.PREV_IS_MOD.ordinal]

                res[TrainCol.PREV_DUR.ordinal] =
                    parts[InputTrainCol.PREV_UP_TIME.ordinal] - parts[InputTrainCol.PREV_DOWN_TIME.ordinal]

                res[TrainCol.PREV_UP_TH_DOWN_DUR.ordinal] =
                    parts[InputTrainCol.TH_DOWN_TIME.ordinal] - parts[InputTrainCol.PREV_UP_TIME.ordinal]

                res[TrainCol.TH_DOWN_NEXT_DOWN_DUR.ordinal] =
                    parts[InputTrainCol.NEXT_DOWN_TIME.ordinal] - parts[InputTrainCol.TH_DOWN_TIME.ordinal]

                res[TrainCol.NEXT_DOWN_TIME.ordinal] = parts[InputTrainCol.NEXT_DOWN_TIME.ordinal]
                res[TrainCol.NEXT_UP_TIME.ordinal] = parts[InputTrainCol.NEXT_UP_TIME.ordinal]
                res[TrainCol.TH_UP_TIME.ordinal] = parts[InputTrainCol.TH_UP_TIME.ordinal]
                res[TrainCol.IS_MOD.ordinal] = parts[InputTrainCol.IS_MOD.ordinal]

                res[TrainCol.NEXT_DUR.ordinal] =
                    parts[InputTrainCol.NEXT_UP_TIME.ordinal] - parts[InputTrainCol.NEXT_DOWN_TIME.ordinal]

                res[TrainCol.TH_DOWN_NEXT_UP_DUR.ordinal] =
                    parts[InputTrainCol.NEXT_UP_TIME.ordinal] - parts[InputTrainCol.TH_DOWN_TIME.ordinal]

                res[TrainCol.LAST_DOWN_TIME.ordinal] = parts[InputTrainCol.LAST_DOWN_TIME.ordinal]

                res[TrainCol.NEXT_DOWN_LAST_DOWN_DUR.ordinal] =
                    parts[InputTrainCol.LAST_DOWN_TIME.ordinal] - parts[InputTrainCol.NEXT_DOWN_TIME.ordinal]

                for ((i, r) in res.withIndex()) {
                    if (r < minValues[i]) {
                        minValues[i] = r
                    }
                    if (r > maxValues[i]) {
                        maxValues[i] = r
                    }

                    if (TrainCol.entries[i].isDur) {
                        res[i] = r.coerceIn(-MAX_VALUE, MAX_VALUE)
                    }
                }

                assert(
                    res[TrainCol.PREV_DUR.ordinal] >= 0 &&
                            res[TrainCol.TH_DOWN_NEXT_DOWN_DUR.ordinal] >= 0 &&
                            res[TrainCol.NEXT_DUR.ordinal] >= 0 &&
                            res[TrainCol.TH_DOWN_NEXT_UP_DUR.ordinal] >= 0 &&
                            res[TrainCol.NEXT_DOWN_LAST_DOWN_DUR.ordinal] >= 0
                ) {
                    "durations must be >= 0, but this was not: ${res.contentToString()}"
                }

                res
            }
            .filter {
                when (MODE) {
                    Mode.ALL -> true
                    Mode.WRAPPED -> it.get { TrainCol.NEXT_UP_TIME } < it.get { TrainCol.TH_UP_TIME }
                    Mode.TRIPLE_DOWN -> it.get { TrainCol.LAST_DOWN_TIME } < min(
                        it.get { TrainCol.NEXT_UP_TIME },
                        it.get { TrainCol.TH_UP_TIME }
                    )// has to be before both -> other cases might be a wrap
                }
            }
            .toMutableList()

        println("Loaded $count events, of which ${result.size} matched the current mode.\n")
        println("These are the durations BEFORE filtering and coercing:\n")
        println("# minimums\n${trainColDurationsToString(minValues)}\n")
        println("# maximums\n${trainColDurationsToString(maxValues)}\n")

        val n = (result.size - NUM_VALIDATION_PER_TYPE).coerceAtMost(NUM_TRAINING_DATA)
        if (USE_ALL_FOR_VALIDATION_DATA) {
            validationData = result.toMutableList()
            result.limitToSample(n)
        } else {
            validationData = result.limitToSampleAndGetRemainder(n)
            if (validationData.size > NUM_VALIDATION_PER_TYPE * 2) {
                validationData = validationData.sample(intArrayOf(NUM_VALIDATION_PER_TYPE, NUM_VALIDATION_PER_TYPE)) {
                    it[TrainCol.IS_MOD.ordinal].toInt()
                }
            }
        }

        return result
    }
}

private fun trainColDurationsToString(minValues: Array<Double>): String {
    return TrainCol.entries.filter { it.isDur }.joinToString("\n") {
        " - " + it.name.lowercase() + " = " + minValues[it.ordinal].toLong().toString()
    }
}


private fun guessIfIsHold(trainingEvent: Array<Double>, progResult: Double): Double {
    // is_mod is 1.0 for true, 0.0 for false
    val isMod = trainingEvent.get { TrainCol.IS_MOD } != 0.0

    val absProgResult = progResult.absoluteValue

    if (isMod == (absProgResult > 0.5)) {
        return 0.0
    }

    val delta = (if (isMod) 0.5 - absProgResult else absProgResult - 0.5).coerceAtMost(MAX_MS_CONSIDERED)

    // to ensure that, even if a solution did not succeed with every item of the training data,
    // the fitness values here would still not sum up to 1
    return 1 + delta / (MAX_MS_CONSIDERED * trainingData.size * 2)
}


private fun estimateOverlap(trainingEvent: Array<Double>, progResult: Double): Double {
    // The overlap between the tap hold and next key have to be larger than
    // the estimate for it to be considered held. Otherwise, it is estimated
    // to be a tap. The overlap is the time from the next press to either
    // the next or tap hold release, whichever comes first.
    var res = progResult.absoluteValue
    if (FIX_MAX_OVERLAP_MS > 0) {
        res = res.coerceAtMost(FIX_MAX_OVERLAP_MS)
    }

    val nextDownTime = trainingEvent.get { TrainCol.NEXT_DOWN_TIME }
    val estimateReleaseHappensAfterIfMod = nextDownTime + res
    val overlapEnd = min(
        trainingEvent.get { TrainCol.NEXT_UP_TIME },
        trainingEvent.get { TrainCol.TH_UP_TIME }
    )

    val estimatesThatIsMod = estimateReleaseHappensAfterIfMod < overlapEnd

    // is_mod is 1.0 for true, 0.0 for false
    val isMod = trainingEvent.get { TrainCol.IS_MOD } != 0.0

    if (estimatesThatIsMod == isMod) {
        return 0.0
    }

    // for calculating the delta of incorrect solutions,
    // we want to use the real delta to give the fitness landscape a gradient
    val realEstimateReleaseHappensAfterIfMod = nextDownTime + progResult.absoluteValue
    val delta = abs(overlapEnd - realEstimateReleaseHappensAfterIfMod).coerceAtMost(MAX_MS_CONSIDERED)

    // to ensure that, even if a solution did not succeed with every item of the training data,
    // the fitness values here would still not sum up to 1
    return 1 + delta / (MAX_MS_CONSIDERED * trainingData.size * 2)
}


fun startSymbolicRegressionToFindHeuristicTapHoldFunctions() {
    println("Kept ${trainingData.size} events for training.")
    println("Kept ${validationData.size} events for validation.")
    println()
    println("A program can use the following variables: $vars")
    println()

    processInitialPopulation()

    val engineBuilder = Engine
        .builder(regression)
        .minimizing()
        .populationSize(1000)
        .survivorsSelector(
            EliteSelector(
                // Number of the best individuals preserved for the next generation: elites
                2,  // Selector used for selecting rest of population.
                TournamentSelector(3)
            )
        )

    val numMutator = TreeRewriteAlterer<Op<Double>, ProgramGene<Double>, Double>(
        GaussianValTreeRewriter(0.8, -50_000.0, 50_000.0, 5.0),
        0.3
    )

    if (ONLY_MUTATE_NUMBERS) {
        engineBuilder.alterers(numMutator)
    } else {
        engineBuilder.alterers(
            SingleNodeCrossover(0.2),
            Mutator(0.2),
            MathRewriteAlterer(0.1),
            numMutator
        )
    }

    val engine = engineBuilder.build()

    val evolutionStream = if (START_FRESH || initialPopulation.isEmpty())
        engine.stream()
    else
        engine.stream(initialPopulation)

    evolutionStream
        .peek(::update)
        .collect(EvolutionResult.toBestEvolutionResult())
}

fun processInitialPopulation() {
    if (initialPopulation.isEmpty()) return

    val initial = mutableListOf<Pair<Double, TreeNode<Op<Double>>>>()
    for ((i, genes) in initialPopulation.withIndex()) {
        if (!genes.isValid) {
            println("$i was invalid\n")
            continue
        }

        val tree = genes.toRewrittenTreeNode()

        val fitness = trainingData.getFitness(tree)
        if (!fitness.isFinite()) {
            println("Fitness is $fitness: $tree")
            continue
        }

        initial.add(fitness to tree)
    }

    initial.sortByDescending { it.first } // smaller = better
    if (initial.size > 10) {
        initial.subList(0, initial.size - 10).clear()
    }

    for (pair in initial) {
        printSolution(pair.second, pair.first)
    }

    val best = initial.last()

    startFindingBestInitialTrainColValue(
        best.second.copy(),
        FINDING_BEST_INITIAL_TRAIN_COL,
        FINDING_BEST_INITIAL_TRAIN_COL_FROM,
        FINDING_BEST_INITIAL_TRAIN_COL_TO
    )

    startHillClimbingIfActive(best.second)

    println("=".repeat(80))
    println()
}


private fun Genotype<ProgramGene<Double>>.toRewrittenTreeNode(): TreeNode<Op<Double>> {
    val tree = this.gene().toTreeNode()
    MathExpr.rewrite(tree)
    return tree
}


fun startFindingBestInitialTrainColValue(
    best: TreeNode<Op<Double>>,
    trainCol: TrainCol,
    from: Int,
    to: Int
) {
    if (!START_FINDING_BEST_INITIAL_TRAIN_COL_VALUE) return
    val name = trainCol.name.lowercase()

    println("Looking for the best initial value of $name")

    if (trainCol != TrainCol.PREV_DUR) {
        val prevDurNode = best.firstOrNull {
            it.value() is Var<Double> && it.value().name() == TrainCol.PREV_DUR.name.lowercase()
        } ?: return

        // The best value that was found using a large sample of the training data
        prevDurNode.value(Const.of(116.0))
    }

    val node = best.firstOrNull {
        it.value() is Var<Double> && it.value().name() == name
    } ?: return

    var bestI = 0
    if (USE_BINARY_SEARCH_TO_FIND_BEST_INITIAL_TRAIN_COL_VALUE) {
        bestI = (from..to).binarySearchForMinimum {
            node.value(Const.of(it.toDouble()))
            trainingData.getFitness(best)
        }
    } else {
        var bestFitness = Double.POSITIVE_INFINITY
        for (i in from..to step 100) {
            node.value(Const.of(i.toDouble()))
            val fitness = trainingData.getFitness(best)
            if (fitness < bestFitness) {
                bestFitness = fitness
                bestI = i
            }
        }
    }

    println("$name = $bestI had the best fitness")
}


fun IntRange.binarySearchForMinimum(continueOnEqual: Boolean = true, get: (Int) -> Double): Int {
    // given that the fitness space is not sorted, this will not actually find the minimum
    if (isEmpty()) throw IllegalArgumentException("the range is empty")

    var low = first
    var high = last
    var lowVal = get(low)
    var mid = 0

    while (low <= high) {
        mid = (low + high).ushr(1)
        val midVal = get(mid)

        if (lowVal > midVal) {
            low = mid + 1
            lowVal = get(low)
            high = mid - 1
        } else if (continueOnEqual || lowVal < midVal) {
            high = mid - 1
        } else {
            break
        }
    }
    return mid
}


fun startHillClimbingIfActive(best: TreeNode<Op<Double>>) {
    if (!START_NUMBER_HILL_CLIMBING) return

    println()
    println("Start hill climbing.")
    println()
    val nodes = best.filter { it.value() is Val<Double> }
    val initialWeights = nodes.map { (it.value() as Val<Double>).value() }.toDoubleArray()

    hillClimb(
        initialWeights,
        whenGoodSolutionFound = { _, fitness, tries -> printHillClimbedSolution(best, fitness, tries) }
    ) { mutations, weights ->
        for ((i, n) in nodes.withIndex()) {
            if (mutations[i] != 0.0) {
                n.value(Const.of(weights[i]))
            }
        }
        trainingData.getFitness(best)
    }
}


fun printHillClimbedSolution(solution: TreeNode<Op<Double>>, fitness: Double, tries: Int) {
    printComment("Tries:       $tries")
    printSolution(solution, fitness)
}


private fun List<Array<Double>>.getFitness(tree: TreeNode<Op<Double>>): Double {
    var m = 0
    var nm = 0
    var mc = 0
    var nmc = 0

    val loss = sumOf {
        val progResult = tree.callWith(it)
        val r = if (MODE == Mode.ALL) {
            estimateOverlap(it, progResult)
        } else {
            guessIfIsHold(it, progResult)
        }

        val isMod = it[TrainCol.IS_MOD.ordinal] > 0
        if (isMod) m += 1 else nm += 1
        if (r == 0.0) {
            if (isMod) mc += 1 else nmc += 1
        }
        r
    }
    printComment("mod: $mc / $m, non-mod: $nmc / $nm")
    return loss + loss * complexity.apply(tree)
}


private fun printResult(
    result: EvolutionResult<ProgramGene<Double>, Double>,
    phenotype: Phenotype<ProgramGene<Double>, Double> = result.bestPhenotype()
) {
    val program = phenotype.genotype().gene()

    val tree: TreeNode<Op<Double>> = program.toTreeNode()
    MathExpr.rewrite(tree) // Simplify result program.

    printComment("Generations: ${result.totalGenerations()}")
    printSolution(tree, phenotype.fitness())
}


private fun printSolution(
    tree: TreeNode<Op<Double>>,
    fitness: Double,
) {
    val complex = complexity.apply(tree)
    val trainingSize = if (UNSTABLE_SAMPLING) UNSTABLE_SAMPLING_SIZE else trainingData.size
    val approxCorrect = getApproxCorrectPercentage(fitness, complex, trainingSize)

    val validationFitness = validationData.getFitness(tree)
    val vApproxCorrect = getApproxCorrectPercentage(validationFitness, complex, validationData.size)
    val diff = approxCorrect.differenceTo(vApproxCorrect)
    printComment("Time:        ${Instant.now()}")
    printComment("Fitness:     ${fitness.toRoundedString()}")
    if (diff > 0.01) {
        val vs = validationData.size.toStringWithThousandSeparator()
        printComment(
            "~V. Correct: ${vApproxCorrect.toRoundedString()} % " +
                    "(of $vs, diff: ${(diff * 100).toRoundedString()} %)"
        )
    }

    val ts = trainingSize.toStringWithThousandSeparator()
    printComment("~Correct:    ${approxCorrect.toRoundedString()} % (of $ts)")
    printComment("Node Count:  ${tree.size()}")


    // fitness is nearly equal to the error here (apart from the complexity factor)
    //println("Error:       " + regression.error(tree))

    printFunction(tree)
    println()
}

fun printFunction(tree: TreeNode<Op<Double>>) {
    if (PRINT_FOR_JAVA) {
        println(FUNCTION_LINE_PREFIX + "parseToProgGenotypeIfMode(Mode.$MODE, \"${MathExpr(tree)}\"),")
    } else {
        printComment("Mode:        $MODE")
        println("${MathExpr(tree)}")
    }
}

fun printComment(comment: String) {
    if (PRINT_FOR_JAVA) {
        println(COMMENT_LINE_PREFIX + comment)
    } else {
        println(comment)
    }
}


fun Double.toRoundedString(decimals: Int = FMT_DECIMAL_PLACES): String {
    return "%.${decimals}f".format(Locale.US, this)
}

fun Int.toStringWithThousandSeparator(): String {
    return "%,d".format(Locale.US, this)
}


fun Double.differenceTo(other: Double): Double {
    if (!this.isFinite() || !other.isFinite()) return 0.0

    val mi = min(this, other)
    val ma = max(this, other)

    // if both are zero, the result will be zero
    // if not, we simply use the diffN
    return if (mi == 0.0) ma else ma / mi - 1
}

private fun getApproxCorrectPercentage(fitness: Double, complexity: Double, dataSize: Int): Double {
    if (!fitness.isFinite()) return fitness;

    // the error function is the fitness function here,
    // and that one is calculated with loss + loss*complexity
    // which is equivalent to loss*(1+complexity)
    // so fitness = loss * (1 + complexity)
    // and we know fitness and complexity, so we turn this into
    val loss = fitness / (1 + complexity)

    return 100 * (dataSize - loss) / dataSize
}

var bestFitness: Double? = null

fun update(result: EvolutionResult<ProgramGene<Double>, Double>) {
    val nf = result.bestPhenotype().fitness()
    val best = bestFitness

    if (best == null || best > nf) {
        bestFitness = nf
        printResult(result)
    }
}