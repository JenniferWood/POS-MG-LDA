package jean.nlp.lda.main;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jean.nlp.lda.assist.FileUtil;
import jean.nlp.lda.distribution.GammaFunction;
import jean.nlp.lda.main.Documents.Document;
import jean.nlp.lda.main.Documents.Document.Sentence;
import jean.nlp.lda.main.LDAModel.parameterName;

public class LDAModel1 {
	Documents docset;
	int D, W; // 文档个数、词汇个数

	int T = 3;// 每个窗口包含三个句子

	float alphaG, alphaL, beta, gamma; // 狄利克雷参数
	float[] alphaM = new float[2];
	int KG, KL;// 两种评论的个数
	int iterations;// 最大迭代次数
	int saveSteps, beginSaveStep;

	/*
	 * 待训练的数组们
	 */
	double[][] thetaG = new double[D][KG]; // θ d->global
	double[][][] thetaL = new double[D][][];// θ d,v->loc
	double[][] phi = new double[KG + KL][W];// φ r=gl/loc,z∈KG+KL->w
	double[][][] psi = new double[D][][]; // ψ d,s->v
	double[][] pai = new double[D][];// π d,v->r

	/*
	 * 计算用的数组们
	 */
	// 计算phi
	int[][] nzw; // (KG+KL)*W the number of times word w appeared in
					// Gtopic/Ltopic
	int[] nrz; // KG+KL the number of words in each topic

	// 计算psi
	int[][][] ndsv;// D*Sd*Vd
	int[][] nds;// D*Sd the length of document d,sentence s

	// 计算pai
	int[][][] ndvr;// D*Vd*2
	int[][] ndv;// D*Vd

	// 计算theta
	int[] ndgl; // D
	int[][] ndglz;// D*KG
	int[][] ndvloc;// D*Vd
	int[][][] ndvlocz;// D*Vd*KL

	/*
	 * 标记用的数组们
	 */
	int[][][] v, w, r, z; // D*Sd*Ws 指示矩阵中每个位置的词分别的 词、种类、主题

	// int[][] v;// D*Sd 指示矩阵中每个位置的句子属于的窗口号

	public LDAModel1(String parameterFile) {
		ArrayList<String> lines = new ArrayList<String>();
		FileUtil.readLines(parameterFile, lines);
		for (String line : lines) {
			String[] parameterKV = line.split("\t");
			switch (parameterName.valueOf(parameterKV[0])) {
			case alphaG:
				alphaG = Float.valueOf(parameterKV[1]);
				System.out.println("alphaG = " + alphaG);
				break;
			case alphaL:
				alphaL = Float.valueOf(parameterKV[1]);
				System.out.println("alphaL = " + alphaL);
				break;
			case beta:
				beta = Float.valueOf(parameterKV[1]);
				System.out.println("beta = " + beta);
				break;
			case gamma:
				gamma = Float.valueOf(parameterKV[1]);
				System.out.println("gamma = " + gamma);
				break;
			case globalTopicNum:
				KG = Integer.parseInt(parameterKV[1]);
				System.out.println("KG = " + KG);
				break;
			case localTopicNum:
				KL = Integer.parseInt(parameterKV[1]);
				System.out.println("KL = " + KL);
				break;
			case iteration:
				iterations = Integer.parseInt(parameterKV[1]);
				System.out.println("iterations = " + iterations);
				break;
			case saveStep:
				saveSteps = Integer.parseInt(parameterKV[1]);
				System.out.println("saveSteps = " + saveSteps);
				break;
			case beginSaveIters:
				beginSaveStep = Integer.parseInt(parameterKV[1]);
				System.out.println("beginSaveStep = " + beginSaveStep);
				break;
			}
		}
		alphaM[0] = (float)KG/(KG+KL);
		alphaM[1] = 1-alphaM[0];
	}

	/*
	 * 初始化
	 */
	public void initialize(Documents docset) {
		this.docset = docset;
		D = docset.docs.size();
		W = docset.wordDict.size();

		nzw = new int[KG + KL][W];
		nrz = new int[KG + KL];
		ndgl = new int[D];
		ndglz = new int[D][KG];

		nds = new int[D][];
		ndsv = new int[D][][];
		ndvr = new int[D][][];
		ndv = new int[D][];
		ndvloc = new int[D][];
		ndvlocz = new int[D][][];

		w = new int[D][][];
		r = new int[D][][];
		z = new int[D][][];
		v = new int[D][][];

		thetaG = new double[D][KG]; // θ d->global
		thetaL = new double[D][][];// θ d,v->loc
		phi = new double[KG + KL][W];// φ r=gl/loc,z∈KG+KL->w
		psi = new double[D][][]; // ψ d,s->v
		pai = new double[D][];// π d,v->r
		
		for (int d = 0; d < D; d++) {
			Sentence[] sents = docset.docs.get(d).docSents;
			int Sd = sents.length;
			int Vd = Sd - T + 1;
			if (Vd <= 0)
				Vd = 1;

			nds[d] = new int[Sd];
			ndsv[d] = new int[Sd][Vd];
			ndvr[d] = new int[Vd][2];
			ndv[d] = new int[Vd];
			ndvloc[d] = new int[Vd];
			ndvlocz[d] = new int[Vd][KL];

			v[d] = new int[Sd][];
			w[d] = new int[Sd][];
			r[d] = new int[Sd][];
			z[d] = new int[Sd][];

			thetaL[d] = new double[Vd][KL];
			psi[d] = new double[Sd][Vd];
			pai[d] = new double[Vd];
			
			for (int s = 0; s < Sd; s++) {
				int[] sentWords = sents[s].sentWords;

				nds[d][s] = sentWords.length;

				w[d][s] = sentWords;
				v[d][s] = new int[sentWords.length];
				r[d][s] = new int[sentWords.length];
				z[d][s] = new int[sentWords.length];

				for (int wo = 0; wo < sentWords.length; wo++) {
					int randv = (int) (Math.random() * Vd);
					v[d][s][wo] = randv; // 为句子中的词随机分配窗口
					ndsv[d][s][randv]++;
					ndv[d][randv]++;

					int randr = (int) (Math.random() * 2); // 为每个词随机分配种类
					r[d][s][wo] = randr;
					ndvr[d][randv][randr]++;

					if (randr == 0) {// global
						int randz = (int) (Math.random() * KG);// 随机分配global主题
						z[d][s][wo] = randz;

						ndgl[d]++;
						ndglz[d][randz]++;

						nzw[randz][wo]++;
						nrz[randz]++;

						//alphaM[0]++;
					} else {
						int randz = (int) (Math.random() * KL);// 随机分配local主题
						z[d][s][wo] = randz;

						ndvloc[d][randv]++;
						ndvlocz[d][randv][randz]++;

						nzw[KG + randz][wo]++;
						nrz[KG + randz]++;

						//alphaM[1]++;
					}
				}
			}

		}
		System.out.println();
	}

	/*
	 * 对文章中的每个句子重新分配窗口，对窗口中的每个词重新分配种类和主题
	 */
	public void inference() {
		for (int i = 0; i < iterations; i++) {
			System.out.println("Iteration " + i);
			if ((i >= beginSaveStep) && (i - beginSaveStep) % saveSteps == 0) {
				System.out.println("中途保存");
				getRequiredMatrixes(i);
				// System.out.println();
				SaveModel(i);
			}else if(i%100==0){
				getRequiredMatrixes(i);
			}

			//getRequiredMatrixes();
			// re-sample v r z
			for (int d = 0; d < D; d++) {
				Sentence[] sents = docset.docs.get(d).docSents;
				int S = sents.length;
				for (int s = 0; s < S; s++) {
					int Ws = sents[s].sentWords.length;
					for (int wo = 0; wo < Ws; wo++)
						sampleVRZ(d, s, wo);
				}
			}
		}
		System.out.println("\nStep3: Done");
		SaveModel(iterations);
		System.exit(0);
	}

	private void SaveModel(int iter) {
		// TODO Auto-generated method stub
		ArrayList<String> global = new ArrayList<String>();
		ArrayList<String> local = new ArrayList<String>();
		
		Map<Integer,Map<String,Double>> topics = new HashMap<Integer,Map<String,Double>>();
		for(int i=0;i<KG+KL;i++){
			Map<String,Double> map = new HashMap<String,Double>();
			topics.put(i, map);
		}
		
		for (int d = 0; d < D; d++) {
			Sentence[] sents = docset.docs.get(d).docSents;
			int S = sents.length;
			for (int s = 0; s < S; s++) {
				int Ws = sents[s].sentWords.length;
				for (int wo = 0; wo < Ws; wo++){
					int zi = z[d][s][wo];
					int wi = w[d][s][wo];
					int ri = r[d][s][wo];
					
					if(ri==1){
						zi += KG;
					}
					topics.get(zi).put(docset.wordDict.get(wi), phi[zi][wi]);
				}
			}
		}
		
		int k = 0;
		for(Map<String, Double> hm:topics.values()){
			StringBuilder sb = new StringBuilder();
			if(k<KG) 
				sb.append(k+":\t");
			else sb.append((k-KG)+":\t");
			getTopWords(hm,sb,10);
			String sbstr = sb.toString();
			sbstr = sbstr.substring(0, sbstr.length() - 1);
			sbstr += "\n";
			if(k<KG) global.add(sbstr);
			else local.add(sbstr);
			k++;
		}
		FileUtil.writeLines("data/LdaResult/lda_"+iter+".gwords", global);
		FileUtil.writeLines("data/LdaResult/lda_"+iter+".lwords", local);
	}



	private void getTopWords(Map<String, Double> hm, StringBuilder sb,int num) {
		// TODO Auto-generated method stub
		
		List<Map.Entry<String, Double>> infoIds =
			    new ArrayList<Map.Entry<String, Double>>(hm.entrySet());
		//排序
		Collections.sort(infoIds, new Comparator<Map.Entry<String, Double>>() {   
		    public int compare(Map.Entry<String, Double> o1, Map.Entry<String, Double> o2) {      
		        //return (o2.getValue() - o1.getValue()); 
		        return (-1)*(o1.getValue()).compareTo(o2.getValue());
		    }
		}); 
		
		num = Math.min(num, infoIds.size());
		for(int i=0;i<num;i++){
			sb.append(infoIds.get(i).getKey()+" "+infoIds.get(i).getValue()+"\t");	
		}
	}

	private void getRequiredMatrixes(int iter) {
		//GammaFunction gf = new GammaFunction(100);
		double changeTG = 0.0, changeTL = 0.0, changeP = 0.0;
		int TG = 0, TL =0, P =0;

		for (int d = 0; d < D; d++) {
			double[] tmp = getPvrz(d, 0, 0);
			for (int i = 0; i < tmp.length; i++) {
				changeTG += tmp[i] - thetaG[d][i];
				TG++;
			}
			thetaG[d] = tmp;
			Sentence[] sents = docset.docs.get(d).docSents;
			int S = sents.length;
			int V = S - T + 1;
			if (V <= 0)
				V = 1;
			for (int s = 0; s < S; s++) {
				psi[d][s] = getPv(d,s,V);
			}
			for (int v = 0; v < V; v++) {
				pai[d] = getPvr(d,v);
				tmp = getPvrz(d, v, 1);
				for (int i = 0; i < tmp.length; i++) {
					changeTL += tmp[i] - thetaL[d][v][i];
					TL++;
				}
				thetaL[d][v] = tmp;
			}
		}

		if (changeTG < 0)
			changeTG *= (-1);
		changeTG /= TG;
		if (changeTL < 0)
			changeTL *= (-1);
		changeTL /= TL;

		for (int z = 0; z < KG + KL; z++) {
			double[] tmp = new double[W];
			for (int w = 0; w < W; w++) {
				/*
				tmp[w] = gf.getValue(W * beta)
						* gf.getValue(nzw[z][w] + beta)
						/ (gf.getValue(nrz[z] + W * beta) * Math.pow(
								gf.getValue(beta), W));*/
				tmp[w] = (nzw[z][w] + beta)/(nrz[z] + W * beta);
			}
			tmp = normalize(tmp);
			for (int i = 0; i < tmp.length; i++) {
				changeP += tmp[i] - phi[z][i];
				P++;
			}
			phi[z] = tmp;
		}
		if (changeP < 0)
			changeP *= (-1);
		changeP /= P;

		// is converged
		if (changeTG + changeTL + changeP <= 0.03) {
			System.out.println("模型已收敛，无需继续");
			// getRequiredMatrixes();
			SaveModel(iter);
			System.exit(0);
		}
	}

	private void sampleVRZ(int d, int s, int wo) {

		Sentence[] sents = docset.docs.get(d).docSents;
		int S = sents.length;
		int V = S - T + 1;
		if (V <= 0)
			V = 1;

		int oldv = v[d][s][wo];
		ndsv[d][s][oldv]--;
		nds[d][s]--;
		double[] P = getPv(d,s,V);
		int newv = samplefromP(P);
		ndsv[d][s][newv]++;
		nds[d][s]++;
		v[d][s][wo] = newv;

		int oldr = r[d][s][wo];
		ndvr[d][oldv][oldr]--;
		ndv[d][oldv]--;
		//alphaM[oldr]--;
		P = getPvr(d,newv);
		//P[0] = sampleFromBeta(d, V);
		//P[1] = 1.0;
		int newr = samplefromP(P);
		ndvr[d][newv][newr]++;
		ndv[d][newv]++;
		//alphaM[newr]++;
		r[d][s][wo] = newr;

		int oldz = z[d][s][wo];

		if (oldr == 0) {
			ndglz[d][oldz]--;
			ndgl[d]--;

			nzw[oldz][wo]--;
			nrz[oldz]--;
		} else {
			ndvlocz[d][oldv][oldz]--;
			ndvloc[d][oldv]--;

			nzw[KG + oldz][wo]--;
			nrz[KG + oldz]--;
		}
		int newz;
		if (newr == 0) {
			P = getPvrz(d, 0, 0);
			newz = samplefromP(P);

			ndglz[d][newz]++;
			ndgl[d]++;

			nzw[newz][wo]++;
			nrz[newz]++;
		} else {
			P = getPvrz(d, newv, 1);
			newz = samplefromP(P);

			ndvlocz[d][newv][newz]++;
			ndvloc[d][newz]++;

			nzw[KG + newz][wo]++;
			nrz[KG + newz]++;
		}
		z[d][s][wo] = newz;
		// float amG = (float)alphaM[0]/(alphaM[0]+alphaM[1]);
		// Pvr[0] =
		// gf.getValue(1)*gf.getValue(ndvr[d][newv][0]+amG)*gf.getValue(ndvr[d][newv][1]+1-amG)/gf.getValue(ndv[d][newv]+1)/gf.getValue(amG)/gf.getValue(1-amG);
		// Pvr[1] = 1-Pvr[0];
		// Pvr = normalize(Pvr);

		//System.out.println("v:" + oldv + "->" + newv + ", r:" + oldr + "->"
		//		+ newr + ", z:" + oldz + "->" + newz);
	}

	private double[] getPvrz(int d, int v, int type) {
		// TODO Auto-generated method stub
		//GammaFunction gf = new GammaFunction(100);
		int K = (type == 0) ? KG : KL;
		double[] P = new double[K];
		if (type == 0) {
			for (int z = 0; z < K; z++) {
				/*
				P[z] = gf.getValue(K * alphaG)
						* gf.getValue(ndglz[d][z] + alphaG)
						/ (gf.getValue(ndgl[d] + K * alphaG) * Math.pow(
								gf.getValue(alphaG), K));*/
				P[z] = (ndglz[d][z] + alphaG) / (ndgl[d] + K * alphaG);
			}
		} else {
			for (int z = 0; z < K; z++) {
				/*
				P[z] = gf.getValue(K * alphaL)
						* gf.getValue(ndvlocz[d][v][z] + alphaL)
						/ (gf.getValue(ndvloc[d][v] + K * alphaL) * Math.pow(
								gf.getValue(alphaL), K));*/
				P[z] = (ndvlocz[d][v][z] + alphaL)/(ndvloc[d][v] + K * alphaL);
			}
		}
		return normalize(P);
	}

	private double sampleFromBeta(int d, int V) {
		// TODO Auto-generated method stub
		GammaFunction gf = new GammaFunction(100);
		//double amG = (double) alphaM[0] / (alphaM[0] + alphaM[1]);
		double B = 1.0;
		double a = 0, b = 0;
		for (int v = 0; v < V; v++) {
			B *= gf.getValue(ndv[d][v] + 1)
					/ (gf.getValue(ndvr[d][v][0] + alphaM[0]) * gf
							.getValue(ndvr[d][v][1] + alphaM[1]));
			if (!Double.isFinite(B)) {
				System.out.println();
			}
			a += ndvr[d][v][0] - alphaM[1];
			b += ndvr[d][v][1] - alphaM[0];

		}
		if(a<0) a=b;
		if(b<0) b=a;
		
		double mid = (a + 1) / (a + b + 2);
		double M = B * Math.pow(mid, a) * Math.pow(1 - mid, b);

		while (true) {
			double xi = Math.random();

			double acc = B * Math.pow(xi, a) * Math.pow(1 - xi, b) / (M * xi);
			double ui = Math.random();
			if (acc >= ui)
				return xi;
		}
	}

	
	private double[] getPvr(int d, int v) {
		// TODO Auto-generated method stub
		//GammaFunction gf = new GammaFunction(100);
		double[] P = new double[2];
		P[0] = (ndvr[d][v][0]+alphaM[0])/(ndv[d][v]+1);
		P[1] = 1.0;

		// double ratio = Math.pow((double)1/(amG*(1-amG)), V);
		// P[0]*=ratio;
		return P;
	}
	
	private double[] getPv(int d, int s, int V) {
		//GammaFunction gf = new GammaFunction(100);
		double[] P = new double[V];

		for (int v = 0; v < V; v++) {
			P[v] = 1.0;
			/*
			P[v] *= T * gamma * gf.getValue(ndsv[d][s][v] + gamma)
					/ gf.getValue(nds[d][s] + T * gamma) / Math.pow(gamma, T);*/
			P[v] *= (ndsv[d][s][v] + gamma) / (nds[d][s] + T * gamma);
		}
		return normalize(P);
	}

	private int samplefromP(double[] pv) {
		double xi = Math.random();
		int k = 0;
		for (double p : pv) {
			if (p >= xi)
				break;
			k++;
		}
		if (k == pv.length)
			k--;
		return k;
	}

	private double[] normalize(double[] P) {
		// TODO Auto-generated method stub
		double sum = 0.0;
		for (double p : P) {
			sum += p;
		}
		for (int i = 0; i < P.length; i++) {
			P[i] /= sum;
			if (i > 0)
				P[i] += P[i - 1];
		}
		return P;
	}
}