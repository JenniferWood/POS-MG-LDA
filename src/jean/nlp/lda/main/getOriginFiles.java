package jean.nlp.lda.main;

import java.io.File;
import java.util.ArrayList;

import jean.nlp.lda.assist.FileUtil;

public class getOriginFiles {
	public static void main(String[] args){
		ArrayList<String> lines = new ArrayList<String>();
		FileUtil.readLines("/Users/apple/Movies/reviews_Movies_and_TV_5.json", lines, 15);
		
		for(String raw: lines){
			ArrayList<String> output = new ArrayList<String>();
			int ih = 16;
			int it = raw.indexOf("asin")-4;
			int ah = raw.indexOf("asin")+8;
			int at = raw.indexOf("reviewerName")-4;
			int rh = raw.indexOf("reviewText")+14;
			int rt = raw.indexOf("overall")-4;
			
			String id = raw.substring(ih,it);
			String asin = raw.substring(ah,at);
			String review = raw.substring(rh, rt);
			String[] reviewSents = review.split("[,\\.\\?!]");
			for(String rs:reviewSents){
				rs = rs.trim();
				rs = rs.replaceAll("\\s+", "\t");
				output.add(rs+"\n");
			}
			String destDirName = "data/reviews/"+asin+"/";
			File dir = new File(destDirName);
			if(!dir.exists())
				dir.mkdirs();
			FileUtil.writeLines("data/reviews/"+asin+"/"+id+".txt", output);
		}
		//FileUtil.writeLines(file, counts);
	}
}
