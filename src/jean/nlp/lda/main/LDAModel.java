package jean.nlp.lda.main;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import jean.nlp.lda.assist.FileUtil;
import jean.nlp.lda.distribution.Beta;
import jean.nlp.lda.distribution.GammaFunction;
import jean.nlp.lda.main.Documents.Document.Sentence;

public class LDAModel {
	int D, W, T=2, KG, KL;
	float alphaG, alphaL;
	int[] alphaM = new int[2];
	float beta, gamma;
	int iterations, beginSaveStep, saveSteps;

	int[][][] words, v, r, z;

	double[][] gphi, lphi;
	double[][][] gtheta, ltheta;
	double[][] pai;

	int[][] Nglzw; // KG * W
	int[] Nglz; // KG

	int[][] Nloczw;// KL * W
	int[] Nlocz;// KL

	int[][][] Ndsv;// D*S*V
	int[][] Nds;// D*S

	int[][][] Ndvr;// D*V*2
	int[][] Ndv;// D*V

	int[][] Ndglz;// D*KG
	int[] Ndgl;// D
	
	int[][][] Ndvlocz;// D*V*KL
	int[][] Ndvloc;//D*V

	Documents docset;

	public enum parameterName {
		alphaG, alphaL, gamma, beta, globalTopicNum, localTopicNum, iteration, saveStep, beginSaveIters;
	}

	public LDAModel(String parameterFile) {
		ArrayList<String> lines = new ArrayList<String>();
		FileUtil.readLines(parameterFile, lines);
		for (String line : lines) {
			String[] parameterKV = line.split("\t");
			switch (parameterName.valueOf(parameterKV[0])) {
			case alphaG:
				alphaG = Float.valueOf(parameterKV[1]);
				if (alphaG < 0)
					throw new IllegalArgumentException("参数异常:alphaG小于0");
				System.out.println("alphaG = " + alphaG);
				break;
			case alphaL:
				alphaL = Float.valueOf(parameterKV[1]);
				if (alphaL < 0)
					throw new IllegalArgumentException("参数异常:alphaL小于0");
				System.out.println("alphaL = " + alphaL);
				break;
			case beta:
				beta = Float.valueOf(parameterKV[1]);
				if (beta < 0)
					throw new IllegalArgumentException("参数异常:beta小于0");
				System.out.println("beta = " + beta);
				break;
			case gamma:
				gamma = Float.valueOf(parameterKV[1]);
				if (gamma < 0)
					throw new IllegalArgumentException("参数异常:gamma小于0");
				System.out.println("gamma = " + gamma);
				break;
			case globalTopicNum:
				KG = Integer.parseInt(parameterKV[1]);
				if (KG < 0)
					throw new IllegalArgumentException("参数异常:KG小于0");
				System.out.println("KG = " + KG);
				break;
			case localTopicNum:
				KL = Integer.parseInt(parameterKV[1]);
				if (KL < 0)
					throw new IllegalArgumentException("参数异常:KL小于0");
				System.out.println("KL = " + KL);
				break;
			case iteration:
				iterations = Integer.parseInt(parameterKV[1]);
				if (iterations < 0)
					throw new IllegalArgumentException("参数异常:iterations小于0");
				System.out.println("iterations = " + iterations);
				break;
			case saveStep:
				saveSteps = Integer.parseInt(parameterKV[1]);
				if (saveSteps < 0)
					throw new IllegalArgumentException("参数异常:saveSteps小于0");
				System.out.println("saveSteps = " + saveSteps);
				break;
			case beginSaveIters:
				beginSaveStep = Integer.parseInt(parameterKV[1]);
				if (beginSaveStep < 0)
					throw new IllegalArgumentException("参数异常:beginSaveStep小于0");
				System.out.println("beginSaveStep = " + beginSaveStep);
				break;
			}
		}
	}

	public void Initialize(Documents docset) {
		this.docset = docset;

		D = docset.docs.size();
		W = docset.wordDict.size();
		alphaM[0] = 0;
		alphaM[1] = 0;

		Nglzw = new int[KG][W]; // KG * W
		Nglz = new int[KG]; // KG

		Nloczw = new int[KL][W];// KL * W
		Nlocz = new int[KL];// KL

		Ndsv = new int[D][][];// D*S*V
		Nds = new int[D][];// D*S

		Ndvr = new int[D][][];// D*V*2
		Ndv = new int[D][];// D*V

		Ndglz = new int[D][KG];// D*KG
		Ndgl = new int[D];// D
		
		Ndvlocz = new int[D][][];// D*V*KL
		Ndvloc = new int[D][];//D*V

		gtheta = new double[D][][];
		ltheta = new double[D][][];

		gphi = new double[KG][W];
		lphi = new double[KL][W];

		pai = new double[D][];

		words = new int[D][][];// document-word matrix
		// 初始时为每一个词随机分配一个窗口、主题类型、主题
		v = new int[D][][];
		r = new int[D][][];
		z = new int[D][][];
		for (int d = 0; d < D; d++) {
			Sentence[] sents = docset.docs.get(d).docSents;
			int S = sents.length;
			int V = S - T + 1;
			if(V <= 0) V=1; //至少一个窗口
			
			Ndsv[d] = new int[S][V];
			Nds[d] = new int[S];

			Ndvr[d] = new int[V][2];
			Ndv[d] = new int[V];
			
			Ndvlocz[d] = new int[V][KL];
			Ndvloc[d] = new int[V];
			
			gtheta[d] = new double[S][KG];
			ltheta[d] = new double[S][KL];

			// int N = docset.docs.get(d).Wd;
			// phi = new double[D][N][N];

			words[d] = new int[S][];
			v[d] = new int[S][];
			r[d] = new int[S][];
			z[d] = new int[S][];

			int glTerm = 0, locTerm = 0;
			for (int s = 0; s < S; s++) {
				int[] sentW = sents[s].sentWords;
				int N = sentW.length;
				words[d][s] = new int[N];
				r[d][s] = new int[N];
				z[d][s] = new int[N];
				v[d][s] = new int[N];
				
				for (int n = 0; n < sentW.length; n++) {
					words[d][s][n] = sentW[n];
					v[d][s][n] = (int) (Math.random() * V); //随机v
					
					Ndsv[d][s][v[d][s][n]]++;
					Nds[d][s]++;

					r[d][s][n] = (int) (Math.random() * 2);// 0-global,1-local
					Ndvr[d][v[d][s][n]][r[d][s][n]]++;
					Ndv[d][v[d][s][n]]++;

					if (r[d][s][n] == 0) {// global
						alphaM[0] = ++glTerm / KG;
						z[d][s][n] = (int) (Math.random() * KG);
						Nglzw[z[d][s][n]][words[d][s][n]]++;
						Nglz[z[d][s][n]]++;

						Ndglz[d][z[d][s][n]]++;
						Ndgl[d]++;
					} else {
						alphaM[1] = ++locTerm / KL;
						z[d][s][n] = (int) (Math.random() * KL);
						Nloczw[z[d][s][n]][words[d][s][n]]++;
						Nlocz[z[d][s][n]]++;

						Ndvlocz[d][v[d][s][n]][z[d][s][n]]++;
						Ndvloc[d][v[d][s][n]]++;
					}
				}
			}
		}
	}

	public void Inference() {
		for (int i = 0; i < iterations; i++) {
			System.out.println("Iteration " + i);
			if ((i == beginSaveStep) || (i - beginSaveStep) % saveSteps == 0) {
				System.out.println("中途保存");
				getRequiredMatrixes();
				SaveModel();
			}

			// sample v r z
			for (int d = 0; d < D; d++) {
				Sentence[] sents = docset.docs.get(d).docSents;
				int S = sents.length;
				for (int s = 0; s < S; s++) {
					int[] sentW = sents[s].sentWords;
					for (int n = 0; n < sentW.length; n++) {
						sampleVRZ(d, s, n);
					}
				}
			}

		}
	}

	private void SaveModel() {
		// TODO Auto-generated method stub
		// 输出每个主题概率最高的二十个词
		ArrayList<String> output = new ArrayList<String>();

		for (int i = 0; i < KG; i++) {
			StringBuilder sb = new StringBuilder();
			sb.append("G" + i + ":\t");
			List<Map.Entry<Integer, Double>> m_list = getSortedList(gphi[i]);
			for (int k = 0; k < 20; k++) {
				sb.append(docset.wordDict.get(m_list.get(k).getKey())
						+ m_list.get(k).getValue() + "\t");
			}
			String sbstr = sb.toString();
			sbstr = sbstr.substring(0, sbstr.length() - 1);
			sbstr += "\n";
			output.add(sbstr);
		}
		FileUtil.writeLines("data/LdaResult/try.gwords", output);
		
		output = new ArrayList<String>();

		for (int i = 0; i < KL; i++) {
			StringBuilder sb = new StringBuilder();
			sb.append("G" + i + ":\t");
			List<Map.Entry<Integer, Double>> m_list = getSortedList(lphi[i]);
			for (int k = 0; k < 20; k++) {
				sb.append(docset.wordDict.get(m_list.get(k).getKey())
						+ m_list.get(k).getValue() + "\t");
			}
			String sbstr = sb.toString();
			sbstr = sbstr.substring(0, sbstr.length() - 1);
			sbstr += "\n";
			output.add(sbstr);
		}
		FileUtil.writeLines("data/LdaResult/try.lwords", output);
		
		//输出每个主题概率最高的五个句子
		List<List<String>> tsents = new ArrayList<List<String>>();
		for(int i=0;i<KG;i++){
			tsents.add(new ArrayList<String>());
		}
		
		for(int d=0;d<D;d++){
			Sentence[] sents = docset.docs.get(d).docSents;
			int S = sents.length;
			int sentz = 0;
			for(int s=0;s<S;s++){
				double sz = 0;
				for(int z=0;z<KG;z++){
					if(gtheta[d][s][z]>sz){
						sz = gtheta[d][s][z];
						sentz = z;
					}
				}
				tsents.get(sentz).add(sents[s].text);
			}
		}
		System.out.println();
	}


	private List<Map.Entry<Integer, Double>> getSortedList(double[] input){
		Map<Integer, Double> m = new HashMap<Integer, Double>();
		for (int j = 0; j < input.length; j++) {
			m.put(j, input[j]);
		}
		List<Map.Entry<Integer, Double>> m_list = new ArrayList<Map.Entry<Integer, Double>>(
				m.entrySet());
		Collections.sort(m_list,
				new Comparator<Map.Entry<Integer, Double>>() {
					public int compare(Map.Entry<Integer, Double> o1,
							Map.Entry<Integer, Double> o2) {
						if (o2.getValue() == null || o1.getValue() == null) {
							return -1;
						}
						if (o2.getValue().compareTo(o1.getValue()) > 0) {
							return 1;
						} else if (o2.getValue().compareTo(o1.getValue()) == 0) {
							return 0;
						} else {
							return -1;
						}
					}
				});
		return m_list;
	}
	private void getRequiredMatrixes() {
		// TODO Auto-generated method stub
		double[][] gphiTmp = new double[KG][W], lphiTmp = new double[KL][W];

		int K = Math.max(KG, KL);
		for (int z = 0; z < K; z++) {
			for (int w = 0; w < W; w++) {
				if (z < Nglzw.length) {
					gphiTmp[z][w] = gphi[z][w];
					gphi[z][w] = Nglzw[z][w] + beta;
				}
				if (z < Nloczw.length) {
					lphiTmp[z][w] = lphi[z][w];
					lphi[z][w] = Nloczw[z][w] + beta;
				}
			}
		}

		double[][][] gthetaTmp = new double[D][][];
		double[][][] lthetaTmp = new double[D][][];

		for (int d = 0; d < D; d++) {
			Sentence[] sents = docset.docs.get(d).docSents;
			int S = sents.length;
			int V = S / T;

			gthetaTmp[d] = new double[S][KG];
			lthetaTmp[d] = new double[S][KL];

			for (int s = 0; s < S; s++) {
				for (int z = 0; z < K; z++) {
					if (z < Nglzw.length) {
						gthetaTmp[d][s][z] = gtheta[d][s][z];
						gtheta[d][s][z] = 0;
					}
					if (z < Nloczw.length) {
						lthetaTmp[d][s][z] = ltheta[d][s][z];
						ltheta[d][s][z] = 0;
					}
					for (int v = 0; v < V; v++) {
						if (z < Nglzw.length) {
							gtheta[d][s][z] += (Ndsv[d][s][v] + gamma)
									/ (Nds[d][s] + 2 * gamma)
									* (Ndvr[d][v][0] + alphaM[0])
									/ (Ndv[d][v] + alphaM[0] + alphaM[1])
									* (Ndglz[d][z] + alphaG)
									/ (Ndgl[d] + KG * alphaG);
						}
						if (z < Nloczw.length) {
							ltheta[d][s][z] += (Ndsv[d][s][v] + gamma)
									/ (Nds[d][s] + 2 * gamma)
									* (Ndvr[d][v][1] + alphaM[1])
									/ (Ndv[d][v] + alphaM[0] + alphaM[1])
									* (Ndvlocz[d][v][z] + alphaL)
									/ (Ndvloc[d][v] + KL * alphaL);
						}
					}
				}
			}
		}

		// check converged
		if (IsConverged(gphiTmp, lphiTmp, gthetaTmp, lthetaTmp)) {
			System.out.println("模型已收敛，无需继续");
			// getRequiredMatrixes();
			SaveModel();
			System.exit(0);
		}
		;
	}

	/*
	private void sampleVRZ(int d, int s, int n) {
		// TODO Auto-generated method stub
		int oldz = z[d][s][n];
		int oldr = r[d][s][n];
		int oldv = v[d][s][n];
		int w = words[d][s][n];

		if (oldr == 0) {// global topic
			Nglzw[oldz][w]--;
			Nglz[oldz]--;
			Ndvr[d][oldv][0]--;
			Ndv[d][oldv]--;
			Ndglz[d][oldz]--;
			Ndgl[d]--;
		} else {// local
			Nloczw[oldz][w]--;
			Nlocz[oldz]--;
			Ndvr[d][oldv][1]--;
			Ndv[d][oldv]--;

			Ndvlocz[d][oldv][oldz]--;
			Ndvloc[d][oldv]--;
		}
		Ndsv[d][s][oldv]--;
		Nds[d][s]--;

		Sentence[] sents = docset.docs.get(d).docSents;
		int S = sents.length;
		int V = S/T;
		if(V==0) V=1;

		double[][] p = new double[V][KG + KL];
		double[] pvr = new double[V];
		double[] pv = new double[V];

		for (int i = 0; i < V; i++) {
			if (i > 0) {
				pv[i] += pv[i - 1];
			}
			pv[i] += (Ndsv[d][s][i] + gamma) / (Nds[d][s] + 2 * gamma);
			pvr[i] = (float) (Ndvr[d][i][0] + alphaM[0])
					/ (Ndv[d][i] + alphaM[0] + alphaM[1]);
			for (int j = 0; j < KG + KL; j++) {
				if (j < KG) {// global
					p[i][j] = (Ndglz[d][j] + alphaG) / (Ndgl[d] + KG * alphaG);
				} else {// local
					p[i][j] = (Ndvlocz[d][i][j - KG] + alphaG)
							/ (Ndvloc[d][i] + KG * alphaG);
				}
			}
		}
		// choose a new v
		double pvthresh = Math.random() * pv[V - 1];
		int k = 0;
		while (k < V) {
			if (pv[k] > pvthresh)
				break;
			k++;
		}
		if (k == 0)
			k = 1;
		v[d][s][n] = k - 1;
		Ndsv[d][s][k - 1]++;
		Nds[d][s]++;

		// choose a new r and a new z
		double[] pvrz = p[k - 1];
		for (int i = 1; i < KG; i++) {
			pvrz[i] += pvrz[i - 1];
		}
		for (int i = KG + 1; i < KG + KL; i++) {
			pvrz[i] += pvrz[i - 1];
		}

		double rthresh = pvr[k - 1];
		double rrand = Math.random();
		if (rrand <= rthresh) {// global
			r[d][s][n] = 0;
			Ndvr[d][v[d][s][n]][0]++;
			Ndv[d][v[d][s][n]]++;

			double zthresh = Math.random() * pvrz[KG - 1];
			int zz = 0;
			while (zz < KG) {
				if (pvrz[zz] > zthresh)
					break;
				zz++;
			}
			if (zz == 0)
				zz = 1;
			z[d][s][n] = zz - 1;
			Nglzw[zz - 1][w]++;
			Nglz[zz - 1]++;
			Ndglz[d][zz - 1]++;
			Ndgl[d]++;
		} else { // local
			r[d][s][n] = 1;
			Ndvr[d][v[d][s][n]][1]++;
			Ndv[d][v[d][s][n]]++;

			double zthresh = Math.random() * pvrz[KG + KL - 1];
			int zz = 0;
			while (zz < KL) {
				if (pvrz[zz + KG] > zthresh)
					break;
				zz++;
			}
			if (zz == 0)
				zz = 1;
			z[d][s][n] = zz - 1;
			Nloczw[zz - 1][w]++;
			Nlocz[zz - 1]++;
			Ndvlocz[d][v[d][s][n]][zz - 1]++;
		}
	}
*/

	private void sampleVRZ(int d, int s, int n) {
		// TODO Auto-generated method stub
		int oldz = z[d][s][n];
		int oldr = r[d][s][n];
		int oldv = v[d][s][n];
		int w = words[d][s][n];
		String theword = docset.wordDict.get(w);

		if (oldr == 0) {// global topic
			Nglzw[oldz][w]--;
			Nglz[oldz]--;
			Ndvr[d][oldv][0]--;
			Ndv[d][oldv]--;
			Ndglz[d][oldz]--;
			Ndgl[d]--;
			alphaM[0]--;
		} else {// local
			Nloczw[oldz][w]--;
			Nlocz[oldz]--;
			Ndvr[d][oldv][1]--;
			Ndv[d][oldv]--;

			Ndvlocz[d][oldv][oldz]--;
			Ndvloc[d][oldv]--;
			alphaM[1]--;
		}
		Ndsv[d][s][oldv]--;
		Nds[d][s]--;

		Sentence[] sents = docset.docs.get(d).docSents;
		int S = sents.length;
		int V = S/T;
		if(V==0) V=1;

		double[] pv = new double[V];
		for(int v = 0;v<V;v++){
			pv[v] = (Ndsv[d][s][v]+gamma)/(Nds[d][s]+T*gamma);
			if(v>0) pv[v]+=pv[v-1];
		}
		int newv = SamplingNumberFromDistribution(pv);
		Ndsv[d][s][newv]++;
		Nds[d][s]++;
		
		double[] pvr = new double[2];
		pvr[0] = (double)(Ndvr[d][newv][0]+alphaM[0])/(Ndv[d][newv]+alphaM[0]+alphaM[1]);
		if(docset.adjDict.contains(theword)) pvr[0]-= pvr[0]/5;
		pvr[1] = 1.0;
		int newr = SamplingNumberFromDistribution(pvr);
		Ndvr[d][newv][newr]++;
		Ndv[d][newv]++;
		alphaM[newr]++;
		
		int newz = 0;
		if(newr==0){//global
			double[] prvz = new double[KG];
			for(int z=0;z<KG;z++){
				prvz[z] = (Ndglz[d][z]+alphaG)/(Ndgl[d]+KG*alphaG);
				if(z>0) prvz[z] += prvz[z-1];
			}
			newz = SamplingNumberFromDistribution(prvz);
			Nglzw[newz][w]++;
			Nglz[newz]++;
			
			Ndglz[d][newz]++;
			Ndgl[d]++;
		}else{
			double[] prvz = new double[KL];
			for(int z=0;z<KL;z++){
				prvz[z] = (Ndvlocz[d][newv][z]+alphaL)/(Ndvr[d][newv][1]+KL*alphaL);
				if(z>0) prvz[z] += prvz[z-1];
			}
			newz = SamplingNumberFromDistribution(prvz);
			Nloczw[newz][w]++;
			Nlocz[newz]++;
			
			Ndvlocz[d][newv][newz]++;
		}
		//System.out.println("newv="+newv+" newr="+newr+" newz="+newz);
	}

	private int SamplingNumberFromDistribution(double[] p) {
		// TODO Auto-generated method stub
		double min = p[0] ,max = p[p.length-1];
		double random = Math.random()*max;
		int num = 0;
		while(num<p.length && p[num]<random){
			num++;
		}
		return num;
	}

	private boolean IsConverged(double[][] gphiTmp, double[][] lphiTmp,
			double[][][] gthetaTmp, double[][][] lthetaTmp) {
		// TODO Auto-generated method stu
		double gp = 0, lp = 0, gt = 0, lt = 0;
		double[] a = new double[KG * W], b = new double[KG * W];

		int m = 0;
		for (int i = 0; i < KG; i++) {
			for (int j = 0; j < W; j++) {
				a[m] = gphiTmp[i][j];
				b[m++] = gphi[i][j];
			}
		}
		gp = MatrixSimilarity(a, b);

		a = new double[KL * W];
		b = new double[KL * W];

		m = 0;
		for (int i = 0; i < KL; i++) {
			for (int j = 0; j < W; j++) {
				a[m] = lphiTmp[i][j];
				b[m++] = lphi[i][j];
			}
		}
		lp = MatrixSimilarity(a, b);

		int gthelen = 0;
		for (int i = 0; i < gthetaTmp.length; i++) {
			for (int j = 0; j < gthetaTmp[i].length; j++) {
				gthelen += KG;
			}
		}
		a = new double[gthelen];
		b = new double[gthelen];
		m = 0;
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < gthetaTmp[i].length; j++) {
				for (int k = 0; k < KG; k++) {
					a[m] = gthetaTmp[i][j][k];
					b[m++] = gtheta[i][j][k];
				}
			}
		}
		gt = MatrixSimilarity(a, b);

		int lthelen = 0;
		for (int i = 0; i < lthetaTmp.length; i++) {
			for (int j = 0; j < lthetaTmp[i].length; j++) {
				lthelen += lthetaTmp[i][j].length;
			}
		}
		a = new double[lthelen];
		b = new double[lthelen];
		m = 0;
		for (int i = 0; i < D; i++) {
			for (int j = 0; j < gthetaTmp[i].length; j++) {
				for (int k = 0; k < KL; k++) {
					a[m] = lthetaTmp[i][j][k];
					b[m++] = ltheta[i][j][k];
				}
			}
		}
		lt = MatrixSimilarity(a, b);

		if (lp>0.7 && gp>0.7 && gt > 0.9 && lt > 0.9)
			return true;
		return false;
	}

	private double MatrixSimilarity(double[] a, double[] b) {
		// person correlation
		double suma = 0, sumb = 0, sumaa = 0, sumbb = 0, sumab = 0;
		int N = a.length;
		for (int i = 0; i < N; i++) {
			suma += a[i];
			sumb += b[i];

			sumaa += a[i] * a[i];
			sumbb += b[i] * b[i];

			sumab += a[i] * b[i];
		}

		double up = sumab - (suma * sumb) / N;
		double down = Math.sqrt(Math.abs((sumaa - suma * suma / N)
				* (sumbb - sumb * sumb / N)));
		if (down == 0)
			return 0;
		return up / down;
	}
}
