//
// Created by Benjamin Sepanski on 1/5/22.
//

#ifndef BRICK_GENE_UTIL_H
#define BRICK_GENE_UTIL_H

#include <cassert>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
#include <string>
#include <utility>

#include "gene-6d-stencils.h"

struct trial_iter_count {
  int num_warmups, num_iters;
};

/**
 * @brief parse args
 * @param[out] perProcessDomainSize the extent in each dimension of the domain
 * @param[out] numProcsPerDim the number of processes per dimension
 * @param[out] numGhostZones the number of ghost zones to use
 * @param[out] outputFileName the output file to write to
 * @param[out] appendToFile true if the file should be appended to
 * @param[in] in input stream to read from
 *
 * @return the number of iterations, with default 100
 */
trial_iter_count parse_args(std::array<int, RANK> *perProcessDomainSize,
                            std::array<int, RANK> *numProcsPerDim,
                            int *numGhostZones,
                            std::string *outputFileName,
                            bool *appendToFile,
                            std::istream &in);

class CSVDataRecorder {
private:
  std::set<std::string> headers;
  std::vector<std::map<std::string, std::string> > rows;
  std::map<std::string, std::string> defaultValues;

  /**
   * convert s to lower case
   * @param s the string to conver to lower
   * @return the lowercase string
   */
  static std::string toLower(const std::string &s);

  /**
   * Insert value into current row
   * @param colName name of the column
   * @param value the value
   */
  void insertIntoCurrentRow(const std::string &colName, const std::string &value);
public:
  CSVDataRecorder() = default;

  /**
   * Start a new row
   */
  void newRow();

  /**
   * Record avg, min, max, sigma for the provided mpi_stats
   *
   * @param colBaseName base name for columns
   * @package value the mpi statistics to record
   */
  void recordMPIStats(const std::string &colBaseName, const mpi_stats &value);

  /**
   * Record a value in the current row
   * @param colName column name (automatically converted to lower case)
   * @param value the value to record
   * @throws runtime_error if a value has already been recorded for this (row, colName) pair
   * @throws runtime_error if no row has been started yet
   */
  template<typename T>
  void record(const std::string & colName, const T &value) {
    std::ostringstream outStream;
    outStream << value;
    this->insertIntoCurrentRow(colName, outStream.str());
  }

  /**
   * Specify a (column, value) pair to be set for every row
   *
   * @param colName the column name
   * @param value the value to initialize rows with
   */
  void setDefaultValue(const std::string &colName, const std::string &value);

  /**
   * Set a default value a value using stringstream for conversion
   *
   * @see setDefaultValue
   */
  template<typename T, typename = typename std::enable_if<!std::is_same<T, std::string>::value>::type>
  void setDefaultValue(const std::string &colName, T value) {
    std::ostringstream outStream;
    outStream << value;
    this->setDefaultValue(colName, outStream.str());
  }

  /**
   * Unset a default value
   * @param defaultColNameToUnset the default column name to unset
   * @throws error if there is no default value associated to defaultColNameToUnset
   */
  void unsetDefaultValue(const std::string &defaultColNameToUnset);

  /**
   * Read in data from a file
   *
   * @param fileName the file to read from
   * @param separator the file separator
   * @param naString the string to indicate a missing value
   */
  void readFromFile(const std::string& fileName, char separator = ',', const std::string& naString = "NA");

  /**
   * Write the CSV out to a file. Overwrites a file if present.
   *
   * @param fileName the file to write to
   * @param separator the separator to use between columns
   * @param naString the string to use for missing values
   */
  void writeToFile(const std::string& fileName, char separator = ',', const std::string& naString = "NA");
};

#endif // BRICK_GENE_UTIL_H
