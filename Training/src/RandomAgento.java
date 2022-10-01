import Block.Tester;
import TestUtil.StatePrinter;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.LibUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;
import game.Game;
import main.collections.FastArrayList;
import other.AI;
import other.context.Context;
import other.move.Move;
import utils.AIUtils;

import java.util.concurrent.ThreadLocalRandom;

public class RandomAgento extends AI {
    protected int player = -1;
    String analysisReport;
    @Override
    public Move selectAction(Game game, Context context, double maxSeconds, int maxIterations, int maxDepth) {

        FastArrayList<Move> legalMoves = game.moves(context).moves();

        // If we're playing a simultaneous-move game, some of the legal moves may be
        // for different players. Extract only the ones that we can choose.
        if (!game.isAlternatingMoveGame())
            legalMoves = AIUtils.extractMovesForMover(legalMoves, player);
        //analysisReport= StatePrinter.setState2D(context);

        final int r = ThreadLocalRandom.current().nextInt(legalMoves.size());


        return legalMoves.get(r);
    }

    @Override
    public String generateAnalysisReport()
    {

        //Tester net = new Tester(2,0,2,null);
        return "";



    }
    @Override
    public void initAI(final Game game, final int playerID)
    {

        //String defaultEngine = CustomEngine.DEFAULT_ENGINE;

    }
}
