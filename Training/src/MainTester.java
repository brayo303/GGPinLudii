
import Block.Tester;
import TestUtil.StatePrinter;
import ai.djl.engine.Engine;
import ai.djl.pytorch.jni.LibUtils;
import customWrapper.CustomGameWrapper;
import customWrapper.CustomStateWrapper;
import customWrapper.MyWrapper;
import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import utils.RandomAI;


import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MainTester {
    public static void main(String[] args){
        List<String> board = new ArrayList<>();
//        board.add("Board Size/4x4");
//        board.add("Swap Rules/Off");
//        board.add("End Rules/Standard");

        Game game = GameLoader.loadGameFromName("Tic-Tac-Toe.lud");
        CustomGameWrapper gameWrapper= new CustomGameWrapper(game);
        System.out.println(Arrays.toString(gameWrapper.stateTensorChannelNames()));
        System.out.println(Arrays.toString(gameWrapper.stateTensorsShape()));
        System.out.println(gameWrapper.numDistinctActions());

        final List<AI> ais = new ArrayList<AI>();
        ais.add(null);
        ais.add(new RandomAI());
        ais.add(new RandomAI());

        Trial trial = new Trial(game);
        Context context = new Context(game, trial);

        game.start(context);

        for (int p = 1; p < ais.size(); ++p) {
            ais.get(p).initAI(game, p);
        }

        final Model model = context.model();

        while (!context.trial().over()) {
            MyWrapper wrapper = new MyWrapper(game);


            model.startNewStep(context, ais, 1.0);

           System.out.println(context.trial().lastMove());
            float[][][] a=wrapper.toTensor(context);
            for(int i = 0 ; i < a.length ; i++){
                for(int j = 0 ; j < a[i].length; j++){
                    for(int k = 0 ; k < a[i][j].length ; k++){
                        System.out.print(a[i][j][k]);
                    }
                    System.out.println();
                }
                System.out.println("-----");
            }
//            System.out.println("++");
        }
    }


}
