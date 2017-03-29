package jean.nlp.lda.assist;

import java.util.ArrayList;

import jean.nlp.lda.conf.CommonConf;

public class StopWords {
	protected static StopWords m_StopWords = new StopWords();
	protected static ArrayList<String> m_words;
	public StopWords(){
		m_words = new ArrayList<String>();
		FileUtil.readLines(CommonConf.STOP_WORD_LOC, m_words);
		System.out.println("停用词加载完毕");
	}
	
	public static boolean isStopword(String word){
		word = word.replaceAll("/.*", "");
		word = word.trim().toLowerCase();
		return m_words.contains(word);
	}
}
