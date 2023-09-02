
EXECUTABLES = beta_cdf beta_mean beta_mean_var beta_pdf beta_pdf_cli beta_quantile

all: $(EXECUTABLES)

beta_cdf: beta_cdf.cpp
	g++ -std=c++14 $(CPPFLAGS) beta_cdf.cpp -o beta_cdf

beta_mean: beta_mean.cpp
	g++ -std=c++14 $(CPPFLAGS) beta_mean.cpp -o beta_mean

beta_mean_var: beta_mean_var.cpp
	g++ -std=c++17 $(CPPFLAGS) beta_mean_var.cpp -o beta_mean_var

beta_pdf: beta_pdf.cpp
	g++ -std=c++14 $(CPPFLAGS) beta_pdf.cpp -o beta_pdf

beta_pdf_cli: beta_pdf_cli.cpp
	g++ -std=c++14 $(CPPFLAGS) beta_pdf_cli.cpp -o beta_pdf_cli

beta_quantile: beta_quantile.cpp
	g++ -std=c++14 $(CPPFLAGS) beta_quantile.cpp -o beta_quantile

clean:
	rm -rf $(EXECUTABLES)
