//
// Created by Benjamin Sepanski on 1/5/22.
//

#include "util.h"

std::vector<unsigned> read_uint_tuple(std::istream &in, char delim) {
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

std::string CSVDataRecorder::toLower(const std::string &s) {
  std::string lowercaseS = s;
  std::transform(s.begin(), s.end(), lowercaseS.begin(),
                 [](const unsigned char c) -> char { return (char) std::tolower(c); });
  return lowercaseS;
}

void CSVDataRecorder::insertIntoCurrentRow(const std::string &colName, const std::string &value) {
  if (this->rows.empty()) {
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
  for(const auto& defaultValue : defaultValues) {
    initialColToVals.insert(defaultValue);
  }
  this->rows.push_back(initialColToVals);
}

template<>
void CSVDataRecorder::record(const std::string &colName, const bool &value) {
  record(colName, value ? "True" : "False");
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

void CSVDataRecorder::writeToFile(const std::string& fileName, bool append, char separator, const std::string& naString) {
  std::ofstream outFile;
  if(append) {
    outFile.open(fileName, std::ios_base::app);
  } else {
    outFile.open(fileName);
  }
  // write header
  if(!append) {
    bool first = true;
    for (const auto &colName : headers) {
      if (!first)
        outFile << separator;
      first = false;
      outFile << colName;
    }
    outFile << "\n";
  }

  // write rows
  for(auto &colToVals : rows) {
    bool first = true;
    for(const auto& colName : headers) {
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
