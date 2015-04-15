#include<iostream>
#include<fstream>
#include<algorithm>
using namespace std;
struct node
{
	char name[100];
	int id;
	int h;
	float val;
	int ac;
};
bool operator < (node n1, node n2)
{
	return n1.val < n2.val;
}
int main(int argc, char* argv[])
{
	if(argc != 3)
	{
		cerr<<"Give args\n";
		return 0;
	}
	cerr<<"Selecting\n";
	int num = stoi(argv[2]);
	char *fileName = argv[1];
	int n;

	node now[100];
	ifstream fin(fileName);
	fin>>n;
	for(int i=0;i<n;++i)
	{
		fin>>now[i].name;
		fin>>now[i].id;
		fin>>now[i].h;
		fin>>now[i].val;
		if(i<n/2)now[i].ac=1;
		else now[i].ac=-1;
	}
	sort(now, now+n);
	cerr<<"Taking "<<num<<"\n";
	cout<<num<<endl;
	int badcnt=0;
	for(int i=0;i<num;++i)
	{
		int x=n-1-i;
		cout<<now[x].name<<" "<<now[x].id<<endl;
		if(now[x].ac != now[x].id)badcnt++;
	}
	cerr<<"Bad Count: "<<badcnt<<endl;
	fin.close();
}
