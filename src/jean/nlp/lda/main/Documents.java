package jean.nlp.lda.main;

import java.io.File;
import java.util.ArrayList;

import jean.nlp.lda.assist.FileUtil;

public class Documents {
	
	ArrayList<Document> docs;
	ArrayList<String> wordDict;
	ArrayList<String> nounDict;
	ArrayList<String> adjDict;
	
	public Documents()
	{
		docs = new ArrayList<Document>();
		wordDict = new ArrayList<String>();
		nounDict = new ArrayList<String>();
		adjDict = new ArrayList<String>();
	}
	public void readDocs(String docsPath)
	{
		File[] a = new File(docsPath).listFiles();
		for(File docFile : new File(docsPath).listFiles())
		{
			if(docFile.getPath().indexOf("/.")!=-1) continue;
			Document doc = new Document(docFile.getAbsolutePath(), wordDict, nounDict, adjDict);
			//System.out.println("<doc>"+doc.printDoc(indexToTermMap));
			docs.add(doc);
		}
		//System.out.println("WordDict Size is "+wordDict.size()+", including "+nounDict.size()+" nouns and "+adjDict.size()+" adjectives");
	}
	
	public static class Document 
	{	
		Sentence[] docSents;
		int Wd;

		public Document(String docName, ArrayList<String> wordDict,ArrayList<String> nounDict,ArrayList<String> adjDict)
		{
			
			//Read file and initialize word index array
			ArrayList<String> docLines = new ArrayList<String>();
			ArrayList<String> words = new ArrayList<String>();
			FileUtil.readLines(docName, docLines);
			
			int lastlen = 0;
			ArrayList<int[]> sentence=new ArrayList<int[]>();
			for(String line : docLines)
			{
				int[] bg = new int[2];
				bg[0] = lastlen;
				
				FileUtil.tokenizeAndLowerCase(line, words);
				
				if(words.size()==lastlen) continue;
				bg[1] = words.size()-1;
				sentence.add(bg);
				
				lastlen = bg[1]+1;
			}
			this.Wd = lastlen;
			
			this.docSents = new Sentence[sentence.size()];
			for(int i = 0; i < sentence.size(); i++)
			{
				docSents[i]=new Sentence(words, sentence.get(i)[0], sentence.get(i)[1], wordDict, nounDict, adjDict);
				//System.out.println("<sent>"+i+" "+docSents[i].printSent(indexToTermMap));
			}
			//System.out.println("Sentences Num is "+docSents.length);
			words.clear();	
		}
		
		public static class Sentence
		{
			int[] sentWords;
			String text;
			
			public Sentence(ArrayList<String> words,int start, int end, ArrayList<String> wordDict, ArrayList<String> nounDict, ArrayList<String> adjDict)
			{
				//Transfer word to index
				sentWords=new int[end-start+1];
				int si=0;
				for(int i = start; i <=end; i++)
				{
					String word = words.get(i);
					if(!wordDict.contains(word)){
						int newIndex = wordDict.size();
						wordDict.add(word);
						sentWords[si++] = newIndex;
						
						if(whichPOS(word)==0) nounDict.add(word);
						else if(whichPOS(word)==1) adjDict.add(word);
					} else {
						sentWords[si++] = wordDict.indexOf(word);
					}
				}
				text = getSentText(wordDict);
			}
			public int whichPOS(String s){//return 0-n,1-a,2-o
				String pos = s.substring(s.indexOf('/'));
				if(pos.indexOf('n')>=0) return 0;
				if(pos.indexOf('a')>=0) return 1;
				return 2;
			}
			public String getSentText(ArrayList<String> wordDict)
			{
				StringBuffer s=new StringBuffer("");
				for(int i=0;i<sentWords.length;i++){
					String tmp = wordDict.get(sentWords[i]);
					tmp = tmp.substring(0,tmp.indexOf('/'));
					s.append(tmp);
				}
				return s.toString();
			}
		}
	}//Class Document
}//Class Documents
