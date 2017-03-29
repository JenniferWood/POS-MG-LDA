package jean.nlp.lda.distribution;

public class GammaFunction 
{
	private int n;
	private double x;
	public GammaFunction(int n){
		this.n = n;
	}
	public double getValue(double x){
		this.x = x;
		double result = 0;
		double even = 0 ,odd = 0;
		for(int i=1;i<n;i++){
			if(i%2==1) odd+=Y(i);
			else even+=Y(i);
		}
		result += 4*odd + 2*even;
		result /= (3*n);
		return result;
	}
	private double Y(int i) {
		// TODO Auto-generated method stub
		double t = (double)i/n;
		double t2 = Math.pow(t, 2);
		
		double newt = t/(1-t2);
		double extra = (1+t2)/Math.pow(1-t2, 2);
		
		double a = Math.pow(newt, x-1);
		double b = Math.pow(2.718281828459045,-1*newt);
		
		return a*b*extra;
	}
}
