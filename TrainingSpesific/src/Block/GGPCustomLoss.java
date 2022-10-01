package Block;

import TestUtil.StatePrinter;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;


public class GGPCustomLoss extends Loss {
    int classAxis;
//    private Map<String, Float> totalLoss;
    public GGPCustomLoss(String name, int classAxis) {
        super(name);

        this.classAxis=classAxis;
        //totalLoss = new ConcurrentHashMap<>();
    }

    @Override
    public NDArray evaluate(NDList label, NDList prediction) {
        //System.out.println(label);
        //System.out.println(prediction);

        //System.out.println(prediction.get(0).get(0));


//        StatePrinter.off=false;
//        StatePrinter.filename="loss.txt";
        NDArray valueLabel = label.get(1);
//        StatePrinter.logln("+++++++");
//        for(int i = 0 ; i < array.size(0) ; i++) {
//            if(i==0) {
//                StatePrinter.logln("["+Arrays.toString(array.get(i).toFloatArray())+",");
//            }else if(i!=array.size(0)-1){
//                StatePrinter.logln(Arrays.toString(array.get(i).toFloatArray())+",");
//            }
//            else{
//                StatePrinter.logln(Arrays.toString(array.get(i).toFloatArray())+"]");
//            }
//        }
////        StatePrinter.logln("+++++++");
         NDArray valuePred = prediction.get(1);
//        StatePrinter.logln("-------");
//        for(int i = 0 ; i < array2.size(0) ; i++) {
//            if(i==0) {
//                StatePrinter.logln("["+Arrays.toString(array2.get(i).toFloatArray())+",");
//            }else if(i!=array2.size(0)-1){
//                StatePrinter.logln(Arrays.toString(array2.get(i).toFloatArray())+",");
//            }
//            else{
//                StatePrinter.logln(Arrays.toString(array2.get(i).toFloatArray())+"]");
//            }
//        }
//        StatePrinter.logln("-------");


        NDArray mse = (valueLabel.sub(valuePred).square()).mean();
        //StatePrinter.logln("mse:"+mse);

        //System.out.println("pred:"+array.getShape());
//        System.out.println("lab:"+label.get(0).getShape());

        NDArray policyPred = prediction.get(0);

//        StatePrinter.logln("+++++++");
//        for(int i = 0 ; i < pred.size(0) ; i++) {
//            if(i==0) {
//                StatePrinter.logln("["+Arrays.toString(pred.get(i).toFloatArray())+",");
//            }else if(i!=pred.size(0)-1){
//                StatePrinter.logln(Arrays.toString(pred.get(i).toFloatArray())+",");
//            }
//            else{
//                StatePrinter.logln(Arrays.toString(pred.get(i).toFloatArray())+"]");
//            }
//        }
//        StatePrinter.logln("+++++++");
        policyPred = policyPred.logSoftmax(classAxis);
       // System.out.println("p"+pred.get(0));
        NDArray smax;
        NDArray policyLabel = label.get(0);
//        StatePrinter.logln("----");
//        for(int i = 0 ; i < lab.size(0) ; i++) {
//            if(i==0) {
//                StatePrinter.logln("["+Arrays.toString(lab.get(i).toFloatArray())+",");
//            }else if(i!=lab.size(0)-1){
//                StatePrinter.logln(Arrays.toString(lab.get(i).toFloatArray())+",");
//            }
//            else{
//                StatePrinter.logln(Arrays.toString(lab.get(i).toFloatArray())+"]");
//            }
//        }
//        StatePrinter.logln("----");
        smax = policyPred.mul(policyLabel).neg().sum(new int[] {classAxis}, true);
        smax=smax.mean();


//        StatePrinter.logln("loss smax:"+smax);
//        StatePrinter.off=false;
        return mse.add(smax);
    }

}
