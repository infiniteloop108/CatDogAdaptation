#include<iostream>
#include<fstream>
using namespace std;
int main(int argc, char *argv[])
{
	if(argc !=4)
	{
		cout<<"Give testfile, model and output file!\n";
		return 0;
	}
	char *testFile = argv[1];
	char *modelFile = argv[2];
	char *outFile = argv[3];
	cerr<<"Loading model ("<<modelFile<<"):\n";
	float *wt;
	ifstream fin(modelFile);
	int sz;
	fin>>sz;
	wt = new float[sz];
	for(int i=0;i<sz;++i)fin>>wt[i];
	fin.close();
	cerr<<"Model read!\n";
	cerr<<"Testing!\n";
	fin.open(testFile);
	
	ofstream fout(outFile);

	int n;
	fin>>n;
	fout<<n<<endl;
	for(int i=0;i<n;++i)
	{
		char fileName[100];
		int class_id;
		fin>>fileName>>class_id;
		cerr<<"TestCase "<<i+1<<"("<<fileName<<"): ";
		ifstream data(fileName);
		int numRect, dim;
		float *pattern;
		data>>numRect>>dim;
		pattern = new float[numRect * dim];
		for(int i=0;i<numRect*dim;++i)data>>pattern[i];

		double valpos=0, valneg =0, mx = -1.0e10;
		int fclass, fh;
		for(int i=0;i<numRect;++i)
		{
			int start = i*dim;
			valpos=0;
			for(int j=0;j<dim;++j)
				valpos+=(pattern[start+j]*wt[j]);
			valneg=0;
			for(int j=0;j<dim;++j)
				valneg-=(pattern[start+j]*wt[j]);
			if(valpos > mx)
			{
				fclass=1;
				fh=i;
				mx = valpos;
			}
			if(valneg > mx)
			{
				fclass=-1;
				fh=i;
				mx = valneg;
			}
		}
		cerr<<fclass<<" "<<fh<<" "<<mx;
		fout<<fileName<<" "<<fclass<<" "<<fh<<" "<<mx<<endl;
		data.close();
		cerr<<endl;
		free(pattern);
	}
	fout.close();
	free(wt);
	return 0;
}
