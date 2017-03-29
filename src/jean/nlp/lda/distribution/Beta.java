package jean.nlp.lda.distribution;

public class Beta {
	int gl,loc;
	public Beta(int[] a){
		gl = a[0];
		loc = a[1];
		
	}
	
	/*
	public double Density(double x){
		double G1 = GammaFunction.getValue(gl);
		double G2 = GammaFunction.getValue(loc);
		double G3 = GammaFunction.getValue(gl+loc);
		
		double B = G3/(G1*G2);
		
		double result = B*Math.pow(x, gl-1)*Math.pow(1-x, loc-1);
		return result;
	}
	
	public double Choose(){
		double y,betaH,x;
		do{
			x = Math.random();
			y = Density(x);
			betaH = Math.random()*Density((double)gl/(gl+loc));
		}while(y<betaH);
		return x;
	}
	*/
}
