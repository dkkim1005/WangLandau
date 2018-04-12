// Wang & Landau algorithm implemented for the square lattice Ising model 

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <trng/yarn2.hpp>
#include <trng/uniform01_dist.hpp>

#define MIN(a, b) (a<b) ? a:b
#define ROUNDOFF(X) static_cast<int>(floor(X+0.5))

// Randon number generator based on the TRNG4
template<typename T = double>
class TinaRNG
{
public:
	explicit TinaRNG(const unsigned long seed = static_cast<unsigned long>(time(NULL)))
	: ran(seed) {}

	T next() {return u01(ran);}
private:
	trng::yarn2 ran;
	trng::uniform01_dist<T> u01;
};

// energy of the square lattice Ising model
inline double
energy(const std::vector<int>& spin, const std::vector<int>& nnsites, const double h_field)
{
	const int len = spin.size();
	double result = 0;
	for(int i=0; i<len; ++i) {
		result += -spin[i]*(spin[nnsites[4*i]] + spin[nnsites[4*i+1]]
				 + spin[nnsites[4*i+2]] + spin[nnsites[4*i+3]]);
	}
	result /= 2;

	for(auto& s : spin) result += h_field*s;

	return result;
}

// Energy difference between a state with flipped spin at site 'site' and an original state
inline double
delta_energy(const std::vector<int>& spin, const std::vector<int>& nnsites,
	     const int site, const double h_field) {
	return (spin[nnsites[4*site]]+spin[nnsites[4*site+1]]+spin[nnsites[4*site+2]]+spin[nnsites[4*site+3]])
	      *(2*spin[site]) - 2*h_field*spin[site];
}

void
write_dos_file(const std::string filename, const std::vector<unsigned int>& h,
	       const std::vector<double>& g, const std::vector<double>& E_range,
	       const int digits = 15) {
	std::ofstream file(filename);

	if(!file.is_open()) {
		std::cout << " --- Error(file '"
			  << filename << "' is not opened...)!" << std::endl;
	}

	file << std::setprecision(digits);

	const size_t len = g.size();

	for(int i=0; i<len; ++i) {
		if(h[i] > 0) file << E_range[i] << " " << std::setw(15) << g[i] << std::endl;
	}
}

int main(int argc, char* argv[])
{
	TinaRNG<> f;
	if(argc < 4) {
		std::cout << "  syntx : ./WangLandau.x L h_field delE" << std::endl;
		exit(1);
	}
	// L : system size,   h_field : Zeeman field,   delE : energy window
	const int L = std::atoi(argv[1]);
	const double h_field = std::atof(argv[2]), delE = std::atof(argv[3]);

	// minE : ground energy,  len : total # of the energy window,  nsweeps : # of the monte-carlo sampling
	const double minE = -L*L*(2+std::abs(h_field));
	const int len = 2*ROUNDOFF(-minE/delE) + 1; //  -|minE| < E < |minE|
	const size_t nsweeps = (L*L)*1e5;

	auto index_with_E = [&minE, &delE](const double& E)->int{return ROUNDOFF((E-minE)/delE);};
	std::vector<double> E_range(len);
	for(int i=0; i<len; ++i) E_range[i] = minE + delE*i;

	std::cout << " --- nsweeps : " << nsweeps
		  << " (" << L << "^2 X " << nsweeps/(L*L)
		  << ")"<< std::endl;

	std::vector<double> lng(len, 0);     // log(DOS(E))
	std::vector<unsigned int> h(len, 0); // histogram for the DOS(E)

	// x : flatness control parameter(h/<h> > x), tol : cutoff limit of the ln(f)
	double lnf = 1., Ei, Ej, x = 0.95, tol = 1e-8;

	std::vector<int> spin(L*L), nnsites(4*L*L);

	for(int i=0; i<L*L; ++i) spin[i] = (f.next() > 0.5) ? 1 : -1;

	// encoding nearest neighbors spin site
	for(int i=0; i<L; ++i) {
		for(int j=0; j<L; ++j) {
			nnsites[4*(L*i+j)]   = (i==0) ? L*(L-1)+j : L*(i-1)+j; // up
			nnsites[4*(L*i+j)+1] = (i==L-1) ? j : L*(i+1)+j;       // down
			nnsites[4*(L*i+j)+2] = (j==0) ? L*i+L-1 : L*i+j-1;     // left
			nnsites[4*(L*i+j)+3] = (j==L-1) ? L*i : L*i+j+1;       // right
		}
	}
	
	Ei = energy(spin, nnsites, h_field);

	int stage = 1;

	// Wang and Landau algotithm iteration loop
	while(true) {
		std::cout << " ! stage:" << std::setw(3) << (stage++)
			  << ", lnf: " << lnf << std::endl;
		again_flatness :
		int n = 0;
		while(n++ < nsweeps) {
			int site = f.next()*(L*L);
			Ej = Ei + delta_energy(spin, nnsites, site, h_field);
			int index_i = index_with_E(Ei), index_j = index_with_E(Ej);
			double lnp = MIN(lng[index_i] - lng[index_j], 0);

			if(f.next() < std::exp(lnp)) { // accept next movement
				lng[index_j] += lnf, h[index_j] += 1;
				spin[site] *= -1;
				Ei = Ej;
			} else { // decline
				lng[index_i] += lnf, h[index_i] += 1;
			}
		}

		// average of histogram h
		double hbar = std::accumulate(h.begin(), h.end(), 0.)/h.size();
		bool isFlat = true;

		for(auto& hi : h) {
			if(hi != 0 && hi/hbar < x) {
				isFlat = false;
				break;
			}
		}

		if(!isFlat) {
			std::cout << " Flatness condition is not satisfied("
				  << "+ montecarlo sweep)."
				  << std::endl;
			goto again_flatness;
		} else {
			lnf /= 2;
			if(lnf < tol) {
				break;
			} else {
				std::fill(h.begin(), h.end(), 0);
			}
		}
	}
	std::cout << " --- terminate loop!" << std::endl;

	// normalizing DOS and writing a data file
	std::vector<double> g(len);
	double lngmax = (*std::max_element(lng.begin(), lng.end())), Z;

	for(int i=0; i<len; ++i) g[i] = std::exp(lng[i] - lngmax);

	Z = std::accumulate(g.begin(), g.end(), 0.);

	for(auto& gi : g) gi /= Z;

	write_dos_file(("E_g-L" + std::string(argv[1]) + "-h_" + std::string(argv[2]) + ".out"), h, g, E_range);
	
	return 0;
}
