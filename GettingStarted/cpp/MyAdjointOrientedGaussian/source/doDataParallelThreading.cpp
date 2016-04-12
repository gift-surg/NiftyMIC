#include <chrono>

#include "itkDomainThreader.h"
#include "itkThreadedIndexedContainerPartitioner.h"

// We have central nervous system cells of different types.
enum CELL_TYPE {
  NEURON,
  ASTROCYTE,
  OLIGODENDROCYTE
};

// Type to hold our list of cells to count.
typedef std::vector< CELL_TYPE >            CellContainerType;
// Type to hold the count for each CELL_TYPE.
typedef std::map< CELL_TYPE, unsigned int > CellCountType;


// This class performs the multi-threaded cell type counting for the
// CellCounter class, show below.  The CellCounter class is the TAssociate, and
// since this class is declared as a friend, it can access the CellCounter's
// private members to compute the cell type count for the CellCounter.
//
// While the threading class can access its associate's private members, it
// generally should only do so in a read-only manner.  Otherwise, attempting to
// write to the same member from multiple threads will cause race conditions
// and result in erroreous output or crash the program.  For this reason, the
// threading class contains its own data structures that can be written to in
// individual threads.  These data structures are set up in the
// BeforeThreadedExecution method, and the results contained in each data
// structure are collected in AfterThreadedExecution.  In this case, we have
// m_CellCountPerThread whose counts are initialized to zero in
// BeforeThreadedExecution and collected together in AfterThreadedExecution.
//
// All members and methods related to the multi-threaded computation are
// encapsulated in this class.
//
// The class inherits from itk::DomainThreader, which provides common
// functionality and defines the stages of the multi-threaded operation.
//
// The itk::DomainThreader is templated over the type of DomainPartitioner used
// to split up the domain, and type of the associated class.  The domain in
// this case is a range of indices of a std::vector< CELL_TYPE > to process, so
// we use a ThreadedIndexedContainerPartitioner.  Other options for a domains
// defined as an iterator range or an image region are the
// ThreadedIteratorRangePartitioner and the ThreadedImageRegionPartitioner,
// respectively.

template< class TAssociate >
class ComputeCellCountThreader :
  public itk::DomainThreader<
    itk::ThreadedIndexedContainerPartitioner, TAssociate >
{
public:
  // Standard ITK typedefs.
  typedef ComputeCellCountThreader            Self;
  typedef itk::DomainThreader<
    itk::ThreadedIndexedContainerPartitioner,
    TAssociate >                              Superclass;
  typedef itk::SmartPointer< Self >           Pointer;
  typedef itk::SmartPointer< const Self >     ConstPointer;

  // The domain is an index range.
  typedef typename Superclass::DomainType DomainType;

  // This creates the ::New() method for instantiating the class.
  itkNewMacro( Self );

protected:
  // We need a constructor for the itkNewMacro.
  ComputeCellCountThreader() {}

private:
  virtual void BeforeThreadedExecution()
    {
    // Reset the counts for all cell types to zero.
    this->m_Associate->m_CellCount[NEURON]          = 0;
    this->m_Associate->m_CellCount[ASTROCYTE]       = 0;
    this->m_Associate->m_CellCount[OLIGODENDROCYTE] = 0;

    // Resize our per-thread data structures to the number of threads that we
    // are actually going to use.  At this point the number of threads that
    // will be used have already been calculated and are available.  The number
    // of threads used depends on the number of cores or processors available
    // on the current system.  It will also be truncated if, for example, the
    // number of cells in the CellContainer is smaller than the number of cores
    // available.
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    this->m_CellCountPerThread.resize( numberOfThreads );
    for( itk::ThreadIdType ii = 0; ii < numberOfThreads; ++ii )
      {
      this->m_CellCountPerThread[ii][NEURON]          = 0;
      this->m_CellCountPerThread[ii][ASTROCYTE]       = 0;
      this->m_CellCountPerThread[ii][OLIGODENDROCYTE] = 0;
      }
    }

  virtual void ThreadedExecution( const DomainType & subDomain,
                                  const itk::ThreadIdType threadId )
    {
    // Look only at the range of cells by the set of indices in the subDomain.
    for( size_t ii = subDomain[0]; ii <= subDomain[1]; ++ii )
      {
      switch( this->m_Associate->m_Cells[ii] )
        {
      case NEURON:
        // Accumulate in the per thread cell count.
        ++(this->m_CellCountPerThread[threadId][NEURON]);
        break;
      case ASTROCYTE:
        ++(this->m_CellCountPerThread[threadId][ASTROCYTE]);
        break;
      case OLIGODENDROCYTE:
        ++(this->m_CellCountPerThread[threadId][OLIGODENDROCYTE]);
        break;
        }
      }
    }

  virtual void AfterThreadedExecution()
    {
    // Accumulate the cell counts per thread in the associate's total cell
    // count.
    const itk::ThreadIdType numberOfThreads = this->GetNumberOfThreadsUsed();
    for( itk::ThreadIdType ii = 0; ii < numberOfThreads; ++ii )
      {
      this->m_Associate->m_CellCount[NEURON] +=
        this->m_CellCountPerThread[ii][NEURON];

      this->m_Associate->m_CellCount[ASTROCYTE] +=
        this->m_CellCountPerThread[ii][ASTROCYTE];

      this->m_Associate->m_CellCount[OLIGODENDROCYTE] +=
        this->m_CellCountPerThread[ii][OLIGODENDROCYTE];
      }
    }

  std::vector< CellCountType > m_CellCountPerThread;
};


// A class to count the cells.
class CellCounter
{
public:
  typedef CellCounter Self;

  typedef ComputeCellCountThreader< Self > ComputeCellCountThreaderType;

  // Constructor.  Create our Threader class instance.
  CellCounter()
    {
    this->m_ComputeCellCountThreader = ComputeCellCountThreaderType::New();
    }

  // Set the cells we want to count.
  void SetCells( const CellContainerType & cells )
    {
    this->m_Cells.resize( cells.size() );
    for( size_t ii = 0; ii < cells.size(); ++ii )
      {
      this->m_Cells[ii] = cells[ii];
      }
    }

  // Count the cells and return the count of each type.
  const CellCountType & ComputeCellCount()
    {
    ComputeCellCountThreaderType::DomainType completeDomain;
    completeDomain[0] = 0;
    completeDomain[1] = this->m_Cells.size() - 1;
    this->m_ComputeCellCountThreader->Execute( this, completeDomain );
    return this->m_CellCount;
    }

private:
  // Stores the count of each type of cell.
  CellCountType     m_CellCount;
  // Stores the cells to count.
  CellContainerType m_Cells;

  // The ComputeCellCountThreader gets to access m_CellCount and m_Cells as
  // needed.
  friend class ComputeCellCountThreader< Self >;
  ComputeCellCountThreaderType::Pointer m_ComputeCellCountThreader;
};


int main( int, char* [] )
{
  // Our cells.
  static const CELL_TYPE cellsArr[] =
    { NEURON, ASTROCYTE, ASTROCYTE, OLIGODENDROCYTE,
    ASTROCYTE, NEURON, NEURON, ASTROCYTE, ASTROCYTE, OLIGODENDROCYTE };

  CellContainerType cells( cellsArr,
                           cellsArr + sizeof(cellsArr) / sizeof(cellsArr[0]) );

  // Measure time: Start
  auto start = std::chrono::system_clock::now();

  // Count them in a multi-threader way.
  CellCounter cellCounter;
  cellCounter.SetCells( cells );
  const CellCountType multiThreadedCellCount = cellCounter.ComputeCellCount();
  std::cout << "Result of the multi-threaded cell count:\n";
  std::cout << "\tNEURON:          " <<
    (*multiThreadedCellCount.find(NEURON)).second << "\n";
  std::cout << "\tASTROCYTE:       " <<
    (*multiThreadedCellCount.find(ASTROCYTE)).second << "\n";
  std::cout << "\tOLIGODENDROCYTE: " <<
    (*multiThreadedCellCount.find(OLIGODENDROCYTE)).second << "\n";

  // Measure time: Stop
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << "Elapsed time: " << diff.count() << " s" << std::endl;



  // Measure time: Start
  start = std::chrono::system_clock::now();

  // Count them in a single-threaded way.
  CellCountType singleThreadedCellCount;
  singleThreadedCellCount[NEURON] = 0;
  singleThreadedCellCount[ASTROCYTE] = 0;
  singleThreadedCellCount[OLIGODENDROCYTE] = 0;
  for( size_t ii = 0; ii < cells.size(); ++ii )
    {
    switch( cells[ii] )
      {
    case NEURON:
      // Accumulate in the per thread cell count.
      ++(singleThreadedCellCount[NEURON]);
      break;
    case ASTROCYTE:
      ++(singleThreadedCellCount[ASTROCYTE]);
      break;
    case OLIGODENDROCYTE:
      ++(singleThreadedCellCount[OLIGODENDROCYTE]);
      break;
      }
    }
  std::cout << "Result of the single-threaded cell count:\n";
  std::cout << "\tNEURON:          " <<
    (*singleThreadedCellCount.find(NEURON)).second << "\n";
  std::cout << "\tASTROCYTE:       " <<
    (*singleThreadedCellCount.find(ASTROCYTE)).second << "\n";
  std::cout << "\tOLIGODENDROCYTE: " <<
    (*singleThreadedCellCount.find(OLIGODENDROCYTE)).second << "\n";

  // Measure time: Stop
  end = std::chrono::system_clock::now();
  diff = end-start;
  std::cout << "Elapsed time: " << diff.count() << " s" << std::endl;


  // Did we get what was expected?  It is always good to check a multi-threaded
  // implementation against a single-threaded implementation to ensure that it
  // gets the same results.
  if( (*multiThreadedCellCount.find(NEURON)).second !=
        singleThreadedCellCount[NEURON] ||
      (*multiThreadedCellCount.find(ASTROCYTE)).second !=
        singleThreadedCellCount[ASTROCYTE] ||
      (*multiThreadedCellCount.find(OLIGODENDROCYTE)).second !=
        singleThreadedCellCount[OLIGODENDROCYTE] )
    {
    std::cerr << "Error: did not get the same results"
      << "for a single-threaded and multi-threaded calculation." << std::endl;
    return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}