package TestUtil;

import customWrapper.MyWrapper;
import other.context.Context;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;

public class StatePrinter {
	public static String filename="1.txt";
	public static Integer height =null;
	public static boolean off = false;
	public static void printState2D(Context context) {
		 
		System.out.println(context.numContainers());
		System.out.println(context.containers()[0].numSites());
		for(int i = 0 ; i < context.containers()[0].numSites(); ) {
			System.out.print(context.state().containerStates()[0].whoCell(i));
			i++;
			if(i%(height)==0) {
				System.out.println();
			}
		}
	}



	public static String setState2D(Context context) {
		String hasil = "=========\n";

		for(int cid = 0 ; cid < context.numContainers();cid++) {
			for (int i = 0; i < context.containers()[cid].numSites(); ) {
				//int contStartSite =

				hasil+=""+context.state().containerStates()[cid].whoCell(i + context.game().equipment().sitesFrom()[cid]);
				i++;
				if (i % (height) == 0) {
					hasil+="\n";
				}
			}
			hasil+="\n";
		}
		hasil += "=========\n";
		return hasil;
	}
	
	public static void logState2D(Context context) {
		if(!off) {
			height = (int) Math.sqrt(context.containers()[0].numSites());
			try {
				BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true));
				for (int i = 0; i < context.containers()[0].numSites(); ) {
					writer.write(context.state().containerStates()[0].whoCell(i) + "");
					i++;
					if (i % (height) == 0) {
						writer.newLine();
					}
				}
				writer.close();

			} catch (IOException e) {
				System.out.println("An error occurred.");
				e.printStackTrace();
			}
		}
	}
	
	

	
	public static void logln(String s) {
		if(!off) {
			try {
				BufferedWriter writer = new BufferedWriter(new FileWriter(filename, true));
				writer.write(s);
				writer.newLine();
				writer.close();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}

	public static void loglnpiece(Context context){
		MyWrapper wrapper = new MyWrapper(context.game());

		float[][][] a=wrapper.toTensor(context);

		try {
			BufferedWriter writer = new BufferedWriter(new FileWriter(filename,true));
			for(int i = 0 ; i < a[4].length ; i++){
				for(int j = 0 ;  j< a[4][i].length; j++){
					if(a[4][i][j]==1){
						writer.write("1");
					}else if(a[5][i][j]==1){
						writer.write("2");
					}else{
						writer.write("0");
					}
				}

				writer.newLine();
			}
			writer.write("=======");
			writer.newLine();
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


	}

	public static void printlnpiece(Context context){
		MyWrapper wrapper = new MyWrapper(context.game());

		float[][][] a=wrapper.toTensor(context);


			for(int i = 0 ; i < a[4].length ; i++){
				for(int j = 0 ;  j< a[4][i].length; j++){
					if(a[0][i][j]==1){
						System.out.print("1");
					}else if(a[1][i][j]==1){
						System.out.print("2");
					}else{
						System.out.print("0");
					}
				}

				System.out.println();
			}



	}


}
