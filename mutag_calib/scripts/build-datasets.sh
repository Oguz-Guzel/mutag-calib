pocket-coffea build-datasets --cfg datasets/datasets_definitions_DATA_BTagMu_run3.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_VJets_run3.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_QCD_MuEnriched_run3.json -o -rs 'T[123]_(FR|IT|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_TTto4Q_run3.json -o -rs 'T[123]_(FR|IT|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_semileptonic.json -o -rs 'T[123]_(FR|IT|BE|CH|UK)_\w+'
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_fullyhadronic.json -o
pocket-coffea build-datasets --cfg datasets/datasets_definitions_singletop_s-channel.json -o -rs 'T[123]_(FR|IT|BE|CH|UK)_\w+'