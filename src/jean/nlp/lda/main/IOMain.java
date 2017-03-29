package jean.nlp.lda.main;

import jean.nlp.lda.conf.CommonConf;
import jean.nlp.lda.main.Documents.Document;

public class IOMain {
	public static void main(String[] args){
		System.out.println("Step0: Reading Prior Parameters");
		LDAModel lda = new LDAModel(CommonConf.PARAMETER_FILE_LOC);
		
		Documents docset = new Documents();
		docset.readDocs(CommonConf.ORIGIN_FILE_LOC);
		System.out.println("\nStep00: Getting All Text");
		System.out.println("共有"+docset.docs.size()+"个文件,词典大小"+docset.wordDict.size()+",名词词典大小"+docset.nounDict.size()+",形容词词典大小"+docset.adjDict.size());
		int i = 1;
		for(Document d:docset.docs){
			System.out.println("第"+i+"个文件包含句子"+d.docSents.length+"条,词语"+d.Wd+"个");
			i++;
		}
		
		System.out.println("\nStep1: Initialize Model");
		lda.Initialize(docset);
		
		System.out.println("\nStep2: Inference Model");
		lda.Inference();
		
		System.out.println("Done!");
	}
}
