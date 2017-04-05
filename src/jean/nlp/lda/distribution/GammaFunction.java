package jean.nlp.lda.distribution;

public class GammaFunction 
{
	private int n;
	public GammaFunction(int n){
		this.n = n;
	}
	public double getValue(double x){
		if(x>=1 && x<2)
			return decimal(x);
		else if(x<1)
			return decimal(x+1)/x;
		else
			return (x-1)*decimal(x-1);
	}
	private double decimal(double x){//大于等于1小于2的
		double result = 0;
		double even = 0 ,odd = 0;
		for(int i=1;i<n;i++){
			if(i%2==1) odd+=Y(i,x);
			else even+=Y(i,x);
		}
		result += 4*odd + 2*even;
		result /= (3*n);
		return result;
	}
	
	private double Y(int i,double x) {
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
