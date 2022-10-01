import Block.NeuralNetworkManager;
import TestUtil.Debug;
import TestUtil.StatePrinter;
import game.Game;
import other.AI;
import other.GameLoader;
import other.context.Context;
import other.model.Model;
import other.trial.Trial;
import search.mcts.MCTS;
import utils.AIFactory;
import utils.AIUtils;
import utils.LudiiAI;
import utils.RandomAI;

import java.io.File;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;


public class TesterWithLog
{

    //-------------------------------------------------------------------------

    /** Name of game we wish to play */
    //static final String GAME_NAME = "Tic-Tac-Toe.lud";

    /** Number of games to play */
    static final int NUM_GAMES = 100;

    static  String path = "D:\\ExperimentReal\\LudiiWrapper\\";

    static final char flag = 't';

    //-------------------------------------------------------------------------

    /**
     * Constructor
     */
    private TesterWithLog()
    {
        // do not instantiate
    }

    //-------------------------------------------------------------------------

    public static void main(final String[] args)
    {

        final Game game ;
        if(Configuration.option.length()!=0) {
            // load and create game
            String[] split = Configuration.option.split(",");
            List<String> list = new LinkedList<>();
            for(int i = 0 ; i < split.length ; i++){
                list.add(split[i]);
                System.out.println(split[i]);
            }
            game= GameLoader.loadGameFromName(Configuration.gameName, list);
        }else{
            game= GameLoader.loadGameFromName(Configuration.gameName);
        }



//        CustomGameWrapper wrapper = new CustomGameWrapper(game);
//        int outShape = wrapper.numDistinctActions();


        final List<AI> ais = new ArrayList<AI>();
        ais.add(null);




        //put agent here

        ais.add(new MyAgent(100));
        ais.add(new TrainAgent());





        //ais.add(new TrainAgent());
        for (int p = 1; p < ais.size(); ++p) {
            ais.get(p).initAI(game, p);
        }

        int result[] = new int[3];
        List<String> opt = game.getOptions();
        Iterator<String> it = opt.iterator();
        String option = "";
        while(it.hasNext()){
            String[] temp = it.next().split("/");
            for(int i = 0 ; i < temp.length ; i++){
                option+=temp[i];
            }
        }
        path = path+Configuration.gameName+option+"/"+Configuration.architechture+"/"+ais.get(1).friendlyName()+"vs"+ais.get(2).friendlyName()+"/";
        File directory = new File(path);

        if(!directory.exists()) {
            directory.mkdirs();
            for (int gameCounter = 0; gameCounter < NUM_GAMES; ++gameCounter) {
                Trial trial = new Trial(game);
                Context context = new Context(game, trial);

                game.start(context);


                final Model model = context.model();


                while (!context.trial().over()) {
                    if (flag != 't') {
                        StatePrinter.filename = path + gameCounter + ".txt";
                        StatePrinter.logState2D(context);
                    }else{
                        StatePrinter.filename = path + gameCounter + ".txt";
                        StatePrinter.loglnpiece(context);
                    }

//                   StatePrinter.printlnpiece(context);
//                   System.out.println("======");
                    StatePrinter.logln("=====");
                    model.startNewStep(context, ais,1.0);
                    //model.startNewStep(context, ais, 1000,100,1000,0);
                    //model.startNewStep(context, ais, 1000,25,1000,0);
                }

                if (flag != 't') {
                    System.out.println(gameCounter + "" + context.trial().status());
                    StatePrinter.logState2D(context);
                    StatePrinter.logln("Outcome = " + context.trial().status());

                    result[context.trial().status().winner()] += 1;
                }else{
                    System.out.println(gameCounter + "" + context.trial().status());
                    StatePrinter.loglnpiece(context);
                    StatePrinter.logln("Outcome = " + context.trial().status());
                    result[context.trial().status().winner()] += 1;
                }

            }
            System.out.println("p1:" + result[1] + " p2:" + result[2]);
            StatePrinter.filename = path + "overall.txt";
            StatePrinter.logln("p1=" + ais.get(1).friendlyName() + " p2=" + ais.get(2).friendlyName());
            StatePrinter.logln("p1:" + result[1] + " p2:" + result[2]);
            StatePrinter.logln("game:" + Configuration.gameName + Configuration.option);
            StatePrinter.logln("mcts-it:" + Configuration.mcts_it);
            StatePrinter.logln("arch:" + Configuration.architechture);
        }else{
            System.out.println("dir alreadyExist");
        }


    }



}
