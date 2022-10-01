package TestUtil;

import other.context.Context;

public class Debug {
    public static boolean flag = false;


    public static  void println(Object s){
        if(flag==true) {
            System.out.println(s+"");
        }
    }

    public static void printState2D(Context context) {
        int height =3;
        if(flag==true) {
            System.out.println(context.numContainers());
            System.out.println(context.containers()[0].numSites());
            for (int i = 0; i < context.containers()[0].numSites(); ) {
                System.out.print(context.state().containerStates()[0].whoCell(i));
                i++;
                if (i % (height) == 0) {
                    System.out.println();
                }
            }
        }
    }
}
