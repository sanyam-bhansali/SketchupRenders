# Runs inside SketchUp (via -RubyStartup). Exports every scene as a reference PNG
# plus scene state JSON, then force-quits without saving.
require 'json'

module SkpRenderRef
  OUT = ENV['SKP_REF_OUT'] || File.join(Dir.tmpdir, 'skp_ref')

  def self.cam_info(cam)
    {
      eye: cam.eye.to_a.map(&:to_f), target: cam.target.to_a.map(&:to_f), up: cam.up.to_a,
      fov: cam.fov, perspective: cam.perspective?, two_point: (cam.respond_to?(:is_2d?) ? cam.is_2d? : nil),
      center_2d: (cam.respond_to?(:center_2d) ? cam.center_2d.to_a : nil),
      scale_2d: (cam.respond_to?(:scale_2d) ? cam.scale_2d : nil),
      fov_is_height: (cam.respond_to?(:fov_is_height?) ? cam.fov_is_height? : nil),
      aspect_ratio: cam.aspect_ratio
    }
  end

  def self.run
    Dir.mkdir(OUT) unless File.exist?(OUT)
    model = Sketchup.active_model
    view = model.active_view
    model.options['PageOptions']['ShowTransition'] = false rescue nil
    info = { vp_width: view.vpwidth, vp_height: view.vpheight, pages: [] }
    pages = model.pages.to_a
    if pages.empty?
      view.write_image(filename: File.join(OUT, '00_Current_view.png'), width: (900.0 * view.vpwidth / view.vpheight).round, height: 900, antialias: true)
      info[:pages] << { name: 'Current view', camera: cam_info(view.camera) }
    end
    pages.each_with_index do |page, i|
      model.pages.selected_page = page
      view.refresh
      safe = page.name.gsub(/[^A-Za-z0-9_-]+/, '_')
      view.write_image(filename: File.join(OUT, format('%02d_%s.png', i, safe)), width: (900.0 * view.vpwidth / view.vpheight).round, height: 900, antialias: true)
      active_sections = []
      model.entities.grep(Sketchup::SectionPlane).each { |sp| active_sections << sp.get_plane if sp.active? }
      info[:pages] << {
        name: page.name, camera: cam_info(view.camera),
        use_section_planes: page.use_section_planes?,
        hidden_entities: page.hidden_entities.size,
        active_section_planes_root: active_sections,
        active_section_nested: model.active_entities.active_section_plane ? true : false
      }
    end
    File.write(File.join(OUT, 'reference.json'), JSON.pretty_generate(info))
  rescue => e
    File.write(File.join(OUT, 'error.txt'), "#{e.class}: #{e.message}\n#{e.backtrace.join("\n")}")
  ensure
    exit!(0)
  end
end

UI.start_timer(8, false) { SkpRenderRef.run }

