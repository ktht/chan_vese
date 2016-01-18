JOBS = hw1 hw2 project
JOBS_TEST = $(JOBS:%=%.test)
JOBS_CLEAN = $(JOBS:%=%.clean)

all: $(JOBS)

test: $(JOBS_TEST)

clean: $(JOBS_CLEAN)

$(JOBS): %:
	@$(MAKE) -C $*

$(JOBS_TEST): %.test:
	@$(MAKE) -C $* test

$(JOBS_CLEAN): %.clean:
	@$(MAKE) -C $* clean

.PHONY: $(JOBS) $(JOBS_TEST) $(JOBS_CLEAN)

