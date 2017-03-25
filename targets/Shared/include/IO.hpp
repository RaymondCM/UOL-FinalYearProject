#include <iostream>
#include <fstream>
#include <vector>
#include <string>

namespace IO
{
	class Writer
	{
	public:
		Writer(const std::string out)
		{
			this->file_path = out;
		}

		void AddLine(std::string s)
		{
			this->lines.push_back(this->line + s);
			this->line = "";
		}

		template<typename... Args>
		void AddLine(std::string s, Args... args)
		{
			this->line += s + '\t';
			this->AddLine(args...);
		}

		void NewFile(std::string file)
		{
			this->file_path = file;
			this->line = "";
			std::string s = this->lines.at(0);
			this->lines.clear();
			this->lines.push_back(s);
		}

		void Write()
		{
			std::ofstream out(this->file_path);
			std::vector<std::string> rows;

			for (int i = 0; i < this->lines.size(); ++i)
			{
				out << this->lines[i];
				if(i < this->lines.size() - 1)
					out << '\n';
			}

			out.close();
		}
	private:
		std::string file_path;
		std::string line = "";
		std::vector<std::string> lines;
	};
	
}