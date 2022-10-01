package DataStructure;


import other.context.Context;

import java.io.Serializable;
import java.util.Arrays;

public class TrainIter implements Serializable {
    float[][][] tensorState;
    float[] tensorAction;

    int currentPlayer;
    float win;

//    Context context;

    public TrainIter(float[][][] tensorState, float[] tensorAction,int currentPlayer) {
        this.tensorState = tensorState;
        this.tensorAction = tensorAction;
        this.currentPlayer=currentPlayer;

    }

    public void setWin(float win) {
        this.win = win;
    }

    public float getWin() {
        return win;
    }

    public int getCurrentPlayer() {
        return currentPlayer;
    }

    public float[][][] getTensorState() {
        return tensorState;
    }

    public float[] getTensorAction() {
        return tensorAction;
    }

    public void setTensorState(float[][][] tensorState) {
        this.tensorState = tensorState.clone();
    }

    public void setTensorAction(float[] tensorAction) {
        this.tensorAction = tensorAction.clone();
    }

    public String toString(){
        return "player:"+currentPlayer+"\n tesorstate:"+ Arrays.toString(tensorState)+"\n"+"actionstate:"+ Arrays.toString(tensorAction)+"\n"+"win:"+win;
    }

//    public void setDebug(Context context){
//        this.context = context;
//    }
//
//    public Context getDebug(){
//        return context;
//    }
}
