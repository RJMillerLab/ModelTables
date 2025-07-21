import modellake.cli_api as modellake

modellake.download('modelcard')
modellake.download('github')
modellake.download('arxiv')
modellake.extract_table('modelcard')
modellake.extract_table('github')
modellake.extract_table('arxiv')
modellake.quality_control('intra')
modellake.quality_control('inter')
modellake.extract_relatedness('paper')
modellake.table_search('tables/example.csv', method='dense')
modellake.plot_analysis()
modellake.repeat_experiments(method='dense', resource='modelcard', relatedness='paper')


