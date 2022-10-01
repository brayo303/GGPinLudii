package customWrapper;

import game.Game;
import other.context.Context;
import other.move.Move;

public class MyWrapper {
    private float[][][] tensor;
    private int gameSize;
    private int x;
    private int y;
    public MyWrapper(Game game){
        gameSize = game.equipment().totalDefaultSites();
        x = (int)Math.sqrt(gameSize);
        y = gameSize/x;

    }
    public int numDistinctActions(){
        return (gameSize+2);
   }
    public int moveToInt(Move move){

        if(move.isPass()){
            return gameSize;
        }else if(move.isSwap()){
            return gameSize+1;
        }
        else{
            int to = move.to();
            return to;
        }


    }
    public float[][][] toTensor(Context contextin){
        tensor = new float[2][x][y];
        Context context = new Context(contextin);
        for (int i = 0; i < context.containers()[0].numSites();i++) {
            int mover = context.state().containerStates()[0].whoCell(i);
            if(mover!=0) {
                tensor[mover-1][i / y][i % y]=1.0f;
            }
        }
        return tensor;

    }
    public int[] stateTensorsShape(){
        return new int[]{2,x,y};
    }
}
