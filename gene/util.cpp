//
// Created by Benjamin Sepanski on 1/5/22.
//

#include "util.h"

/**
 * @brief Reads a tuple of unsigneds delimited by delim
 *
 * @param in the input stream to read from
 * @param delim the delimiter between unsigneds
 * @return std::vector<unsigned> of the values read in
 */
std::vector<unsigned> read_uint_tuple(std::istream &in, char delim = ',') {
  std::vector<unsigned> tuple;
  unsigned value;
  do {
    if (in.peek() == delim)
      in.get();
    in >> value;
    tuple.push_back(value);
  } while (in.peek() == delim);
  return tuple;
}

trial_iter_count parse_args(std::array<int, RANK> *per_process_domain_size,
                            std::array<int, RANK> *num_procs_per_dim,
                            std::string *outputFileName,
                            bool *appendToFile,
                            std::istream &in) {
  std::string option_string;
  trial_iter_count iter_count;
  iter_count.num_iters = 100;
  iter_count.num_warmups = 5;
  std::vector<unsigned> tuple;
  bool read_dom_size = false, read_num_iters = false, read_num_procs_per_dim = false,
       read_num_warmups = false;
  *outputFileName = "results.csv";
  *appendToFile = false;
  std::string help_string = "Program options\n"
                            "  -h: show help (this message)\n"
                            "  Domain size,  in array order contiguous first\n"
                            "  -d: comma separated Int[6], per-process domain size\n"
                            "  Num Tasks per dimension, in array order contiguous first\n"
                            "  -p: comma separated Int[6], num process per dimension"
                            "  Benchmark control:\n"
                            "  -I: number of iterations, default 100 \n"
                            "  -W: number of warmup iterations, default 5\n"
                            "  -o: csv file to write to (default results.csv)\n"
                            "  -a: If passed, will append data to output file (if it already exists)\n"
                            "Example usage:\n"
                            "  weak/gene6d -d 70,16,24,48,32,2 -p 1,1,3,1,2,1\n";
  std::ostringstream error_stream;
  while (in >> option_string) {
    if (option_string[0] != '-' || option_string.size() != 2) {
      error_stream << "Unrecognized option " << option_string << std::endl;
    }
    if (error_stream.str().size() != 0) {
      error_stream << help_string;
      throw std::runtime_error(error_stream.str());
    }
    switch (option_string[1]) {
    case 'a':
      *appendToFile = true;
      break;
    case 'd':
      tuple = read_uint_tuple(in, ',');
      if (read_dom_size) {
        error_stream << "-d option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        error_stream << "Expected extent of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), per_process_domain_size->begin());
      }
      read_dom_size = true;
      break;
    case 'o':
      in >> *outputFileName;
      break;
    case 'p':
      tuple = read_uint_tuple(in, ',');
      if (read_num_procs_per_dim) {
        error_stream << "-p option should only be passed once" << std::endl;
      } else if (tuple.size() != RANK) {
        error_stream << "Expected num procs per dim of length " << RANK << ", not " << tuple.size();
      } else {
        std::copy(tuple.begin(), tuple.end(), num_procs_per_dim->begin());
      }
      read_num_procs_per_dim = true;
      break;
    case 'I':
      if (read_num_iters) {
        error_stream << "-I option should only be passed once" << std::endl;
      } else {
        in >> iter_count.num_iters;
      }
      read_num_iters = true;
      break;
    case 'W':
      if (read_num_warmups) {
        error_stream << "-W option should only be passed once" << std::endl;
      } else {
        in >> iter_count.num_warmups;
      }
      read_num_warmups = true;
      break;
    default:
      error_stream << "Unrecognized option " << option_string << std::endl;
    }
  }
  if (!read_num_procs_per_dim) {
    error_stream << "Missing -p option" << std::endl << help_string;
    throw std::runtime_error(error_stream.str());
  }
  if (!read_dom_size) {
    error_stream << "Missing -d option" << std::endl << help_string;
    throw std::runtime_error(error_stream.str());
  }
  return iter_count;
}

std::string CSVDataRecorder::toLower(const std::string &s) {
  std::string lowercaseS = s;
  std::transform(s.begin(), s.end(), lowercaseS.begin(),
                 [](const unsigned char c) -> char { return std::tolower(c); });
  return lowercaseS;
}

void CSVDataRecorder::insertIntoCurrentRow(const std::string &colName, const std::string &value) {
  if (this->rows.size() <= 0) {
    throw std::runtime_error("No active row");
  }
  std::string lowercaseColName = toLower(colName);

  // Make sure this (row, col) pair has not been set yet, then
  // insert
  std::map<std::string, std::string> *colToVals = &this->rows.back();
  auto column = colToVals->find(lowercaseColName);
  if (column != colToVals->end()) {
    std::ostringstream errStream;
    errStream << "Value for column " << lowercaseColName << " in current row already set to value "
              << column->second;
    throw std::runtime_error(errStream.str());
  }

  auto insertHandle = colToVals->insert(std::make_pair(lowercaseColName, value));
  assert(insertHandle.second);

  headers.insert(lowercaseColName);
}

void CSVDataRecorder::newRow() {
  std::map<std::string, std::string> initialColToVals;
  for(auto defaultValue : defaultValues) {
    initialColToVals.insert(defaultValue);
  }
  this->rows.push_back(initialColToVals);
}

template<>
void CSVDataRecorder::record(const std::string &colName, const bool &value) {
  record(std::move(colName), value ? "True" : "False");
}

void CSVDataRecorder::recordMPIStats(const std::string &colBaseName, const mpi_stats &value) {
  record(colBaseName + "Avg", value.avg);
  record(colBaseName + "Min", value.min);
  record(colBaseName + "Max", value.max);
  record(colBaseName + "StdDev", value.sigma);
}

void CSVDataRecorder::setDefaultValue(const std::string &colName, const std::string &value) {
  std::string lowercaseColName = toLower(colName);
  headers.insert(lowercaseColName);
  defaultValues[lowercaseColName] = value;
}

void CSVDataRecorder::unsetDefaultValue(const std::string &defaultColNameToUnset) {
  auto defaultValueHandle = defaultValues.find(toLower(defaultColNameToUnset));
  if(defaultValueHandle == defaultValues.end()) {
    std::ostringstream errStream;
    errStream << toLower(defaultColNameToUnset) << " is not associated to any default value";
    throw std::runtime_error(errStream.str());
  }
  defaultValues.erase(defaultValueHandle);
}

void CSVDataRecorder::readFromFile(std::string fileName, char separator,
                                   std::string naString) {
  std::ifstream inFile;
  inFile.open(fileName);
  std::vector<std::string> lines;
  std::string headerString, line;
  bool header = true;
  while(std::getline(inFile, line)) {
    if(header) {
      headerString = line;
      header = false;
    } else {
      lines.push_back(line);
    }
  }
  inFile.close();

  // read in header
  std::vector<std::string> headerValues;
  std::istringstream headerStream(headerString);
  std::string headerValue;
  while(std::getline(headerStream, headerValue, separator)) {
    headerValues.push_back(headerValue);
  }

  // read in data
  for(auto rowString : lines) {
    this->newRow();

    std::istringstream rowStream(rowString);
    std::string rowValue;
    auto columnNameIterator = headerValues.begin();
    while(std::getline(rowStream, rowValue, separator)) {
      if(columnNameIterator == headerValues.end()) {
        throw std::runtime_error("Row is longer than number of headers");
      }

      if(rowValue != naString) {
        this->record(*columnNameIterator, rowValue);
      }
      columnNameIterator++;
    }
  }
}

void CSVDataRecorder::writeToFile(std::string fileName, char separator, std::string naString) {
  std::ofstream outFile;
  outFile.open(fileName);
  // write header
  bool first = true;
  for(auto colName : headers) {
    if(!first) outFile << separator;
    first = false;
    outFile << colName;
  }
  outFile << "\n";

  // write rows
  for(auto &colToVals : rows) {
    first = true;
    for(auto colName : headers) {
      if(!first) outFile << separator;
      first = false;

      auto colVal = colToVals.find(colName);
      if(colVal == colToVals.end()) {
        outFile << naString;
      } else {
        outFile << colVal->second;
      }
    }
    outFile << "\n";
  }

  outFile.close();
}
