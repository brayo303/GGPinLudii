public class Configuration {
    static int argsTrainIter = 100;
    static int argsNumSelfPlay = 25;
    static int argsHistoryIter = 10;
    static int argsEvalCompetition = 20;
    static int argsEpoch = 10;
    static int argsDepthOrFilterNum = 64;
    static String path = "D:/EDir/Test/";
    static int mcts_it = 100;
    static int feature[] =  null;
    static int mcts_it_eval = 50;
    static float winThreshold = 0.55f;
    static int startIteration= 0;
    static float learningRate = 0.001f;
    static int miniBatchSize = 64;
    static String gameName = "Reversi.lud";
    //static String option = "";
    static String option = "Board Size/6x6,Start Rules/Fixed Start";
    //static String option = "Board Size/5x5,Swap Rules/Off,End Rules/Standard";
    static char architechture = 't';
    static int tempThreshold = 50;
    static int skipEval = 0;
    static char optimizerType = 'a';
    static boolean drawAccept = false;
    static Float epsilon = 0.25f;
    static Double alpha = 0.3;
    static Float dropout= null;


}

// [GameName] [Path] [learningRate] [NNDepth] [miniBatchSize] [argsStartIter] [argsTrainIter]  [argsNumSelfPlay] [argsHistoryIter]
// [argsSelfPlayCompetition] [mcts-it] [winThreshold] [tempThreshold]

// Default
// !java -jar LudiiMCTSVer3.1.jar "Tic-Tac-Toe.lud" "D:/TicTacToeLogs/" "0.01" "3" "64" "0" "100" "20" "1000" "50" "100" "0.6" "4"
